//! Water rendering: Gerstner wave animation, dual-layer normal-map scrolling,
//! Fresnel reflection/refraction blending, depth-based coloring, shore foam,
//! and caustics patterns.
//!
//! Pure Rust — all wave and optical math on CPU.

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

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalize(&self) -> Self {
        let l = self.length();
        if l < 1e-10 {
            return Self::new(0.0, 1.0, 0.0);
        }
        Self::new(self.x / l, self.y / l, self.z / l)
    }

    pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub fn add(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    pub fn scale(&self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
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
    pub fn dot(a: &Vec2, b: &Vec2) -> f32 {
        a.x * b.x + a.y * b.y
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

// ── Gerstner wave ────────────────────────────────────────────────

/// Parameters for a single Gerstner wave.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GerstnerWave {
    pub amplitude: f32,
    pub wavelength: f32,
    pub speed: f32,
    pub direction: Vec2,
    pub steepness: f32,
}

impl GerstnerWave {
    pub fn new(amplitude: f32, wavelength: f32, speed: f32, dir_x: f32, dir_z: f32) -> Self {
        let len = (dir_x * dir_x + dir_z * dir_z).sqrt().max(1e-10);
        Self {
            amplitude,
            wavelength: wavelength.max(0.01),
            speed,
            direction: Vec2::new(dir_x / len, dir_z / len),
            steepness: 0.5,
        }
    }

    fn frequency(&self) -> f32 {
        std::f32::consts::TAU / self.wavelength
    }

    fn phase_speed(&self) -> f32 {
        self.speed * self.frequency()
    }

    /// Evaluate the Gerstner displacement at (x, z, time).
    /// Returns (dx, dy, dz) displacement from rest position.
    pub fn evaluate(&self, x: f32, z: f32, time: f32) -> Vec3 {
        let w = self.frequency();
        let pos = Vec2::new(x, z);
        let dot = Vec2::dot(&self.direction, &pos);
        let phase = w * dot - self.phase_speed() * time;
        let cos_p = phase.cos();
        let sin_p = phase.sin();
        let qi = self.steepness / (w * self.amplitude).max(1e-10);
        Vec3::new(
            qi * self.amplitude * self.direction.x * cos_p,
            self.amplitude * sin_p,
            qi * self.amplitude * self.direction.y * cos_p,
        )
    }

    /// Compute tangent and bitangent for normal derivation.
    pub fn tangent_bitangent(&self, x: f32, z: f32, time: f32) -> (Vec3, Vec3) {
        let w = self.frequency();
        let pos = Vec2::new(x, z);
        let dot = Vec2::dot(&self.direction, &pos);
        let phase = w * dot - self.phase_speed() * time;
        let cos_p = phase.cos();
        let sin_p = phase.sin();
        let wa = w * self.amplitude;
        let s = self.steepness;
        let tangent = Vec3::new(
            1.0 - s * self.direction.x * self.direction.x * wa * sin_p,
            self.direction.x * wa * cos_p,
            -s * self.direction.x * self.direction.y * wa * sin_p,
        );
        let bitangent = Vec3::new(
            -s * self.direction.x * self.direction.y * wa * sin_p,
            self.direction.y * wa * cos_p,
            1.0 - s * self.direction.y * self.direction.y * wa * sin_p,
        );
        (tangent, bitangent)
    }
}

// ── Wave system (sum of Gerstner waves) ──────────────────────────

/// Collection of Gerstner waves forming the water surface.
#[derive(Debug, Clone, PartialEq)]
pub struct WaveSystem {
    pub waves: Vec<GerstnerWave>,
    pub base_height: f32,
}

impl WaveSystem {
    pub fn new(base_height: f32) -> Self {
        Self {
            waves: Vec::new(),
            base_height,
        }
    }

    pub fn add_wave(&mut self, wave: GerstnerWave) {
        self.waves.push(wave);
    }

    /// Summed displacement at (x, z, time).
    pub fn displacement(&self, x: f32, z: f32, time: f32) -> Vec3 {
        let mut total = Vec3::new(0.0, self.base_height, 0.0);
        for wave in &self.waves {
            let d = wave.evaluate(x, z, time);
            total = total.add(&d);
        }
        total
    }

    /// Displaced surface position at (x, z, time).
    pub fn surface_position(&self, x: f32, z: f32, time: f32) -> Vec3 {
        let d = self.displacement(x, z, time);
        Vec3::new(x + d.x, d.y, z + d.z)
    }

    /// Surface normal computed from cross product of tangent/bitangent sums.
    pub fn surface_normal(&self, x: f32, z: f32, time: f32) -> Vec3 {
        let mut tangent = Vec3::new(1.0, 0.0, 0.0);
        let mut bitangent = Vec3::new(0.0, 0.0, 1.0);
        for wave in &self.waves {
            let (t, b) = wave.tangent_bitangent(x, z, time);
            tangent = Vec3::new(
                tangent.x + t.x - 1.0,
                tangent.y + t.y,
                tangent.z + t.z,
            );
            bitangent = Vec3::new(
                bitangent.x + b.x,
                bitangent.y + b.y,
                bitangent.z + b.z - 1.0,
            );
        }
        // normal = bitangent x tangent
        let normal = Vec3::new(
            bitangent.y * tangent.z - bitangent.z * tangent.y,
            bitangent.z * tangent.x - bitangent.x * tangent.z,
            bitangent.x * tangent.y - bitangent.y * tangent.x,
        );
        normal.normalize()
    }

    /// Height-only query (ignoring XZ displacement).
    pub fn height_at(&self, x: f32, z: f32, time: f32) -> f32 {
        self.displacement(x, z, time).y
    }
}

// ── Normal map scrolling ─────────────────────────────────────────

/// Dual-layer normal map scroll parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalMapScroll {
    pub layer1_speed: Vec2,
    pub layer2_speed: Vec2,
    pub layer1_scale: f32,
    pub layer2_scale: f32,
}

impl Default for NormalMapScroll {
    fn default() -> Self {
        Self {
            layer1_speed: Vec2::new(0.03, 0.02),
            layer2_speed: Vec2::new(-0.02, 0.04),
            layer1_scale: 1.0,
            layer2_scale: 0.5,
        }
    }
}

impl NormalMapScroll {
    /// Compute the UV offset for each layer at given time.
    pub fn layer1_uv(&self, u: f32, v: f32, time: f32) -> Vec2 {
        Vec2::new(
            u * self.layer1_scale + self.layer1_speed.x * time,
            v * self.layer1_scale + self.layer1_speed.y * time,
        )
    }

    pub fn layer2_uv(&self, u: f32, v: f32, time: f32) -> Vec2 {
        Vec2::new(
            u * self.layer2_scale + self.layer2_speed.x * time,
            v * self.layer2_scale + self.layer2_speed.y * time,
        )
    }

    /// Blend two procedural normal perturbations.
    pub fn blended_normal(&self, u: f32, v: f32, time: f32) -> Vec3 {
        let uv1 = self.layer1_uv(u, v, time);
        let uv2 = self.layer2_uv(u, v, time);
        let n1x = (uv1.x * 7.0).sin() * 0.3;
        let n1z = (uv1.y * 7.0).cos() * 0.3;
        let n2x = (uv2.x * 11.0).sin() * 0.2;
        let n2z = (uv2.y * 11.0).cos() * 0.2;
        Vec3::new(n1x + n2x, 1.0, n1z + n2z).normalize()
    }
}

// ── Fresnel ──────────────────────────────────────────────────────

/// Schlick's Fresnel approximation.
/// `cos_theta`: dot(view, normal), `f0`: reflectance at normal incidence.
pub fn fresnel_schlick(cos_theta: f32, f0: f32) -> f32 {
    let c = (1.0 - cos_theta).clamp(0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    f0 + (1.0 - f0) * c5
}

/// Blend reflection and refraction colors based on Fresnel.
pub fn fresnel_blend(
    reflection: &Color,
    refraction: &Color,
    view_dir: &Vec3,
    normal: &Vec3,
    f0: f32,
) -> Color {
    let cos_theta = Vec3::dot(view_dir, normal).abs().clamp(0.0, 1.0);
    let f = fresnel_schlick(cos_theta, f0);
    reflection.lerp(refraction, 1.0 - f)
}

// ── Depth-based color ────────────────────────────────────────────

/// Water color configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterColorConfig {
    pub shallow_color: Color,
    pub deep_color: Color,
    pub max_visible_depth: f32,
    pub absorption_rate: f32,
}

impl Default for WaterColorConfig {
    fn default() -> Self {
        Self {
            shallow_color: Color::new(0.1, 0.6, 0.7, 0.6),
            deep_color: Color::new(0.02, 0.1, 0.2, 0.95),
            max_visible_depth: 20.0,
            absorption_rate: 0.3,
        }
    }
}

impl WaterColorConfig {
    /// Color at given depth below surface.
    pub fn color_at_depth(&self, depth: f32) -> Color {
        let t = (depth / self.max_visible_depth.max(0.01)).clamp(0.0, 1.0);
        let exp_t = 1.0 - (-self.absorption_rate * depth).exp();
        let blend = t.max(exp_t).clamp(0.0, 1.0);
        self.shallow_color.lerp(&self.deep_color, blend)
    }
}

// ── Foam ─────────────────────────────────────────────────────────

/// Shore foam effect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoamConfig {
    pub depth_threshold: f32,
    pub intensity: f32,
    pub scroll_speed: f32,
    pub foam_color: Color,
}

impl Default for FoamConfig {
    fn default() -> Self {
        Self {
            depth_threshold: 1.5,
            intensity: 0.8,
            scroll_speed: 0.5,
            foam_color: Color::new(0.9, 0.95, 1.0, 0.9),
        }
    }
}

impl FoamConfig {
    /// Foam amount at given water depth (0 = at shore, large = deep).
    pub fn foam_amount(&self, depth: f32) -> f32 {
        if depth > self.depth_threshold {
            return 0.0;
        }
        let t = 1.0 - (depth / self.depth_threshold.max(0.01));
        (t * self.intensity).clamp(0.0, 1.0)
    }

    /// Procedural foam pattern at UV + time.
    pub fn foam_pattern(&self, u: f32, v: f32, time: f32) -> f32 {
        let scrolled_u = u + time * self.scroll_speed;
        let pattern = ((scrolled_u * 5.0).sin() * (v * 7.0).cos()).abs();
        pattern.clamp(0.0, 1.0)
    }
}

// ── Caustics ─────────────────────────────────────────────────────

/// Underwater caustics pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausticsConfig {
    pub scale: f32,
    pub speed: f32,
    pub intensity: f32,
}

impl Default for CausticsConfig {
    fn default() -> Self {
        Self {
            scale: 2.0,
            speed: 0.4,
            intensity: 0.6,
        }
    }
}

impl CausticsConfig {
    /// Sample the caustics brightness at (x, z, time). Returns [0, intensity].
    pub fn sample(&self, x: f32, z: f32, time: f32) -> f32 {
        let sx = x * self.scale;
        let sz = z * self.scale;
        let t = time * self.speed;
        let v1 = ((sx + t).sin() * (sz - t * 0.7).cos()).abs();
        let v2 = ((sx * 1.3 - t * 0.5).cos() * (sz * 0.9 + t * 1.2).sin()).abs();
        let combined = (v1 + v2) * 0.5;
        combined.clamp(0.0, 1.0) * self.intensity
    }
}

// ── Water surface compositor ─────────────────────────────────────

/// Full water surface configuration and evaluation.
#[derive(Debug, Clone)]
pub struct WaterSurface {
    pub waves: WaveSystem,
    pub normal_scroll: NormalMapScroll,
    pub color_config: WaterColorConfig,
    pub foam: FoamConfig,
    pub caustics: CausticsConfig,
    pub f0: f32,
}

impl WaterSurface {
    pub fn new(base_height: f32) -> Self {
        Self {
            waves: WaveSystem::new(base_height),
            normal_scroll: NormalMapScroll::default(),
            color_config: WaterColorConfig::default(),
            foam: FoamConfig::default(),
            caustics: CausticsConfig::default(),
            f0: 0.02,
        }
    }

    /// Add a Gerstner wave to the surface.
    pub fn add_wave(&mut self, wave: GerstnerWave) {
        self.waves.add_wave(wave);
    }

    /// Evaluate the displaced surface position at (x, z, time).
    pub fn surface_position(&self, x: f32, z: f32, time: f32) -> Vec3 {
        self.waves.surface_position(x, z, time)
    }

    /// Surface height (Y) at (x, z, time).
    pub fn height_at(&self, x: f32, z: f32, time: f32) -> f32 {
        self.waves.height_at(x, z, time)
    }

    /// Evaluate the composite water color at a surface point.
    pub fn evaluate_color(
        &self,
        x: f32,
        z: f32,
        time: f32,
        water_depth: f32,
        view_dir: &Vec3,
    ) -> Color {
        let normal = self.waves.surface_normal(x, z, time);
        let base_color = self.color_config.color_at_depth(water_depth);
        // Fresnel: blend with a "sky reflection" approximation
        let sky_color = Color::new(0.5, 0.7, 1.0, 1.0);
        let fresnel_color = fresnel_blend(&sky_color, &base_color, view_dir, &normal, self.f0);
        // Foam
        let foam_amt = self.foam.foam_amount(water_depth);
        let foam_pattern = self.foam.foam_pattern(x, z, time);
        let foam_factor = foam_amt * foam_pattern;
        let with_foam = fresnel_color.lerp(&self.foam.foam_color, foam_factor);
        // Caustics (add brightness)
        let caustic = self.caustics.sample(x, z, time);
        Color::new(
            (with_foam.r + caustic * 0.2).min(1.0),
            (with_foam.g + caustic * 0.2).min(1.0),
            (with_foam.b + caustic * 0.15).min(1.0),
            with_foam.a,
        )
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
    fn gerstner_wave_creates() {
        let w = GerstnerWave::new(1.0, 10.0, 2.0, 1.0, 0.0);
        assert!(approx(w.amplitude, 1.0, 1e-6));
        assert!(approx(w.direction.x, 1.0, 1e-6));
    }

    #[test]
    fn gerstner_wave_displacement() {
        let w = GerstnerWave::new(1.0, 10.0, 1.0, 1.0, 0.0);
        let d = w.evaluate(5.0, 0.0, 0.0);
        assert!(d.y.abs() <= 1.0 + 1e-4, "amplitude capped");
    }

    #[test]
    fn gerstner_wave_varies_time() {
        let w = GerstnerWave::new(1.0, 10.0, 2.0, 1.0, 0.0);
        let d0 = w.evaluate(0.0, 0.0, 0.0);
        let d1 = w.evaluate(0.0, 0.0, 1.0);
        assert!(
            (d0.y - d1.y).abs() > 1e-6,
            "wave should change over time"
        );
    }

    #[test]
    fn wave_system_base_height() {
        let ws = WaveSystem::new(5.0);
        let h = ws.height_at(0.0, 0.0, 0.0);
        assert!(approx(h, 5.0, 1e-6));
    }

    #[test]
    fn wave_system_multi_wave() {
        let mut ws = WaveSystem::new(0.0);
        ws.add_wave(GerstnerWave::new(0.5, 8.0, 1.0, 1.0, 0.0));
        ws.add_wave(GerstnerWave::new(0.3, 4.0, 2.0, 0.0, 1.0));
        let h = ws.height_at(3.0, 3.0, 1.0);
        assert!(h.abs() <= 1.0, "combined height should be within sum of amplitudes");
    }

    #[test]
    fn wave_system_surface_normal() {
        let ws = WaveSystem::new(0.0);
        let n = ws.surface_normal(0.0, 0.0, 0.0);
        // Flat water → normal should be close to (0, 1, 0)
        assert!(n.y > 0.9);
    }

    #[test]
    fn wave_system_surface_position() {
        let ws = WaveSystem::new(3.0);
        let p = ws.surface_position(5.0, 5.0, 0.0);
        assert!(approx(p.y, 3.0, 1e-4));
    }

    #[test]
    fn normal_map_scroll_default() {
        let ns = NormalMapScroll::default();
        assert!(ns.layer1_scale > 0.0);
    }

    #[test]
    fn normal_map_scroll_uv_changes() {
        let ns = NormalMapScroll::default();
        let uv1_a = ns.layer1_uv(0.5, 0.5, 0.0);
        let uv1_b = ns.layer1_uv(0.5, 0.5, 10.0);
        assert!((uv1_a.x - uv1_b.x).abs() > 0.01);
    }

    #[test]
    fn normal_map_blended_normal_unit() {
        let ns = NormalMapScroll::default();
        let n = ns.blended_normal(1.0, 2.0, 0.5);
        let len = n.length();
        assert!(approx(len, 1.0, 1e-4));
    }

    #[test]
    fn fresnel_schlick_at_normal() {
        let f = fresnel_schlick(1.0, 0.02);
        assert!(approx(f, 0.02, 1e-4));
    }

    #[test]
    fn fresnel_schlick_at_grazing() {
        let f = fresnel_schlick(0.0, 0.02);
        assert!(approx(f, 1.0, 1e-4));
    }

    #[test]
    fn fresnel_blend_color() {
        let refl = Color::new(1.0, 1.0, 1.0, 1.0);
        let refr = Color::new(0.0, 0.0, 0.0, 1.0);
        let view = Vec3::new(0.0, 1.0, 0.0);
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let c = fresnel_blend(&refl, &refr, &view, &normal, 0.02);
        // At normal incidence, mostly refraction
        assert!(c.r < 0.5);
    }

    #[test]
    fn water_color_shallow() {
        let cfg = WaterColorConfig::default();
        let c = cfg.color_at_depth(0.0);
        assert!(approx(c.r, cfg.shallow_color.r, 1e-4));
    }

    #[test]
    fn water_color_deep() {
        let cfg = WaterColorConfig::default();
        let c = cfg.color_at_depth(100.0);
        assert!(approx(c.r, cfg.deep_color.r, 1e-2));
    }

    #[test]
    fn foam_amount_at_shore() {
        let foam = FoamConfig::default();
        let amt = foam.foam_amount(0.0);
        assert!(amt > 0.5);
    }

    #[test]
    fn foam_amount_deep() {
        let foam = FoamConfig::default();
        let amt = foam.foam_amount(10.0);
        assert!(approx(amt, 0.0, 1e-6));
    }

    #[test]
    fn foam_pattern_bounded() {
        let foam = FoamConfig::default();
        for i in 0..20 {
            let p = foam.foam_pattern(i as f32 * 0.5, i as f32 * 0.3, 1.0);
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn caustics_sample_bounded() {
        let c = CausticsConfig::default();
        for i in 0..20 {
            let v = c.sample(i as f32 * 0.7, i as f32 * 1.1, 2.0);
            assert!(v >= 0.0 && v <= c.intensity + 1e-6);
        }
    }

    #[test]
    fn water_surface_creation() {
        let ws = WaterSurface::new(0.0);
        assert_eq!(ws.waves.waves.len(), 0);
        assert!(approx(ws.f0, 0.02, 1e-6));
    }

    #[test]
    fn water_surface_with_waves() {
        let mut ws = WaterSurface::new(0.0);
        ws.add_wave(GerstnerWave::new(0.5, 6.0, 1.5, 1.0, 0.5));
        let h = ws.height_at(3.0, 2.0, 1.0);
        assert!(h.abs() <= 1.0);
    }

    #[test]
    fn water_surface_evaluate_color() {
        let mut ws = WaterSurface::new(0.0);
        ws.add_wave(GerstnerWave::new(0.3, 8.0, 1.0, 1.0, 0.0));
        let view = Vec3::new(0.0, 1.0, 0.0);
        let c = ws.evaluate_color(5.0, 5.0, 1.0, 2.0, &view);
        assert!(c.r >= 0.0 && c.r <= 1.0);
        assert!(c.g >= 0.0 && c.g <= 1.0);
        assert!(c.b >= 0.0 && c.b <= 1.0);
    }

    #[test]
    fn water_surface_shallow_has_foam() {
        let ws = WaterSurface::new(0.0);
        let view = Vec3::new(0.0, 1.0, 0.0);
        let c_shallow = ws.evaluate_color(0.0, 0.0, 0.0, 0.1, &view);
        let c_deep = ws.evaluate_color(0.0, 0.0, 0.0, 10.0, &view);
        // Shallow should be brighter from foam
        assert!(c_shallow.r >= c_deep.r - 0.1 || c_shallow.g >= c_deep.g - 0.1);
    }
}
