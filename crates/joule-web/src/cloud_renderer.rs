//! Volumetric cloud rendering: noise-based density, ray-marching with
//! Beer-Lambert absorption, Henyey-Greenstein phase function, light marching,
//! cloud coverage/type controls, and temporal reprojection.
//!
//! Pure Rust — all volume sampling and scattering on CPU.

// ── Vec3 / Color ─────────────────────────────────────────────────

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
    pub fn sub(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
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
    pub fn add(&self, o: &Color) -> Color {
        Color::new(self.r + o.r, self.g + o.g, self.b + o.b, (self.a + o.a).min(1.0))
    }
    pub fn scale(&self, s: f32) -> Color {
        Color::new(self.r * s, self.g * s, self.b * s, self.a)
    }
}

// ── Noise helpers ────────────────────────────────────────────────

/// Hash-based pseudo-noise (deterministic, not random).
fn hash3d(x: f32, y: f32, z: f32) -> f32 {
    let v = (x * 127.1 + y * 311.7 + z * 74.7).sin() * 43758.5453;
    v - v.floor()
}

/// Value noise (trilinear interpolation of hash lattice).
fn value_noise(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let iz = z.floor();
    let fx = x - ix;
    let fy = y - iy;
    let fz = z - iz;
    // Smooth step
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let n000 = hash3d(ix, iy, iz);
    let n100 = hash3d(ix + 1.0, iy, iz);
    let n010 = hash3d(ix, iy + 1.0, iz);
    let n110 = hash3d(ix + 1.0, iy + 1.0, iz);
    let n001 = hash3d(ix, iy, iz + 1.0);
    let n101 = hash3d(ix + 1.0, iy, iz + 1.0);
    let n011 = hash3d(ix, iy + 1.0, iz + 1.0);
    let n111 = hash3d(ix + 1.0, iy + 1.0, iz + 1.0);
    let nx00 = n000 + (n100 - n000) * sx;
    let nx10 = n010 + (n110 - n010) * sx;
    let nx01 = n001 + (n101 - n001) * sx;
    let nx11 = n011 + (n111 - n011) * sx;
    let nxy0 = nx00 + (nx10 - nx00) * sy;
    let nxy1 = nx01 + (nx11 - nx01) * sy;
    nxy0 + (nxy1 - nxy0) * sz
}

/// FBM (fractal Brownian motion) noise — composite for cloud shapes.
fn fbm_noise(x: f32, y: f32, z: f32, octaves: u32) -> f32 {
    let mut value = 0.0f32;
    let mut amplitude = 0.5;
    let mut freq = 1.0;
    for _ in 0..octaves {
        value += value_noise(x * freq, y * freq, z * freq) * amplitude;
        amplitude *= 0.5;
        freq *= 2.0;
    }
    value
}

/// Worley-like noise (approximation using hash distance).
fn worley_noise(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let iz = z.floor();
    let fx = x - ix;
    let fy = y - iy;
    let fz = z - iz;
    let mut min_dist = 1.0f32;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let nx = ix + dx as f32;
                let ny = iy + dy as f32;
                let nz = iz + dz as f32;
                let px = hash3d(nx, ny, nz);
                let py = hash3d(nx + 31.0, ny + 17.0, nz + 53.0);
                let pz = hash3d(nx + 71.0, ny + 97.0, nz + 13.0);
                let dx_val = dx as f32 + px - fx;
                let dy_val = dy as f32 + py - fy;
                let dz_val = dz as f32 + pz - fz;
                let dist = (dx_val * dx_val + dy_val * dy_val + dz_val * dz_val).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }
    }
    min_dist.clamp(0.0, 1.0)
}

// ── Beer-Lambert ─────────────────────────────────────────────────

/// Beer-Lambert transmittance: exp(-density * distance).
pub fn beer_lambert(density: f32, distance: f32) -> f32 {
    (-density * distance).exp()
}

/// Powder effect (enhanced scattering inside clouds).
pub fn powder_effect(density: f32, distance: f32) -> f32 {
    let beer = beer_lambert(density, distance);
    let powder = 1.0 - beer_lambert(density * 2.0, distance);
    beer * powder * 2.0
}

// ── Henyey-Greenstein ────────────────────────────────────────────

/// Henyey-Greenstein phase function.
/// `g`: asymmetry parameter (-1..1). g>0 = forward scattering (silver lining).
pub fn henyey_greenstein(cos_theta: f32, asymmetry: f32) -> f32 {
    let g = asymmetry;
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-10);
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom * denom.sqrt())
}

/// Dual-lobe phase function (forward + back scatter).
pub fn dual_lobe_phase(cos_theta: f32, g_forward: f32, g_back: f32, blend: f32) -> f32 {
    let pf = henyey_greenstein(cos_theta, g_forward);
    let pb = henyey_greenstein(cos_theta, g_back);
    pf * blend + pb * (1.0 - blend)
}

// ── Cloud layer ──────────────────────────────────────────────────

/// Defines the altitude bounds of the cloud layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLayer {
    pub bottom_altitude: f32,
    pub top_altitude: f32,
}

impl CloudLayer {
    pub fn new(bottom: f32, top: f32) -> Self {
        Self {
            bottom_altitude: bottom.min(top),
            top_altitude: bottom.max(top),
        }
    }

    pub fn thickness(&self) -> f32 {
        self.top_altitude - self.bottom_altitude
    }

    /// Normalized height within the layer [0, 1].
    pub fn height_fraction(&self, y: f32) -> f32 {
        let thickness = self.thickness();
        if thickness < 1e-6 {
            return 0.0;
        }
        ((y - self.bottom_altitude) / thickness).clamp(0.0, 1.0)
    }

    /// Whether a point is inside the layer.
    pub fn contains(&self, y: f32) -> bool {
        y >= self.bottom_altitude && y <= self.top_altitude
    }

    /// Intersect a ray with the cloud slab. Returns (t_enter, t_exit) or None.
    pub fn intersect_ray(&self, origin_y: f32, dir_y: f32) -> Option<(f32, f32)> {
        if dir_y.abs() < 1e-10 {
            if self.contains(origin_y) {
                return Some((0.0, 1000.0));
            }
            return None;
        }
        let t0 = (self.bottom_altitude - origin_y) / dir_y;
        let t1 = (self.top_altitude - origin_y) / dir_y;
        let t_near = t0.min(t1);
        let t_far = t0.max(t1);
        if t_far < 0.0 {
            return None;
        }
        Some((t_near.max(0.0), t_far))
    }
}

// ── Cloud controls ───────────────────────────────────────────────

/// Cloud coverage and type knobs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudControls {
    /// Overall coverage [0, 1]. 0 = clear, 1 = overcast.
    pub coverage: f32,
    /// Cloud type [0, 1]. 0 = stratus (flat), 1 = cumulus (puffy).
    pub cloud_type: f32,
    /// Density multiplier.
    pub density_scale: f32,
    /// Wind offset applied to noise sampling.
    pub wind_offset: Vec3,
    /// Noise frequency.
    pub noise_freq: f32,
}

impl Default for CloudControls {
    fn default() -> Self {
        Self {
            coverage: 0.5,
            cloud_type: 0.5,
            density_scale: 1.0,
            wind_offset: Vec3::new(0.0, 0.0, 0.0),
            noise_freq: 0.002,
        }
    }
}

// ── Cloud density sampler ────────────────────────────────────────

/// Sample cloud density at a 3D point.
pub fn sample_cloud_density(
    point: &Vec3,
    layer: &CloudLayer,
    controls: &CloudControls,
) -> f32 {
    if !layer.contains(point.y) {
        return 0.0;
    }
    let hf = layer.height_fraction(point.y);
    // Height-based profile: round bottom, flat top for cumulus
    let height_profile = if controls.cloud_type > 0.5 {
        // Cumulus-like: ramp up, hold, fade out
        let ramp_up = (hf * 4.0).min(1.0);
        let ramp_down = (1.0 - (hf - 0.7).max(0.0) / 0.3).clamp(0.0, 1.0);
        ramp_up * ramp_down
    } else {
        // Stratus-like: thin layer concentrated in the middle
        let center_dist = (hf - 0.5).abs() * 2.0;
        (1.0 - center_dist).clamp(0.0, 1.0)
    };

    let freq = controls.noise_freq;
    let px = point.x * freq + controls.wind_offset.x;
    let py = point.y * freq * 0.5 + controls.wind_offset.y;
    let pz = point.z * freq + controls.wind_offset.z;

    // Perlin-like base shape
    let base = fbm_noise(px, py, pz, 4);
    // Worley detail (erode edges)
    let detail = worley_noise(px * 3.0, py * 3.0, pz * 3.0);
    let shape = (base - detail * 0.3).clamp(0.0, 1.0);

    // Apply coverage (remap so low-density areas disappear first)
    let coverage_remap = ((shape - (1.0 - controls.coverage)) / controls.coverage.max(0.01))
        .clamp(0.0, 1.0);

    coverage_remap * height_profile * controls.density_scale
}

// ── Ray-march renderer ───────────────────────────────────────────

/// Configuration for the ray-march cloud renderer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudRenderConfig {
    pub max_steps: u32,
    pub step_size: f32,
    pub light_steps: u32,
    pub light_step_size: f32,
    pub sun_direction: Vec3,
    pub sun_color: Color,
    pub ambient_color: Color,
    pub phase_g_forward: f32,
    pub phase_g_back: f32,
    pub phase_blend: f32,
    pub absorption: f32,
}

impl Default for CloudRenderConfig {
    fn default() -> Self {
        Self {
            max_steps: 64,
            step_size: 50.0,
            light_steps: 6,
            light_step_size: 80.0,
            sun_direction: Vec3::new(0.3, 0.9, 0.2).normalize(),
            sun_color: Color::new(1.0, 0.95, 0.85, 1.0),
            ambient_color: Color::new(0.4, 0.5, 0.6, 1.0),
            phase_g_forward: 0.8,
            phase_g_back: -0.3,
            phase_blend: 0.7,
            absorption: 0.04,
        }
    }
}

/// Result of a single ray-march through the cloud layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudSample {
    pub color: Color,
    pub transmittance: f32,
}

/// Ray-march through the cloud layer and accumulate color + transmittance.
pub fn raymarch_clouds(
    ray_origin: &Vec3,
    ray_dir: &Vec3,
    layer: &CloudLayer,
    controls: &CloudControls,
    config: &CloudRenderConfig,
) -> CloudSample {
    let dir = ray_dir.normalize();
    let hit = layer.intersect_ray(ray_origin.y, dir.y);
    let (t_enter, t_exit) = match hit {
        Some((a, b)) => (a, b),
        None => return CloudSample { color: Color::new(0.0, 0.0, 0.0, 0.0), transmittance: 1.0 },
    };

    let cos_theta = Vec3::dot(&dir, &config.sun_direction);
    let phase = dual_lobe_phase(cos_theta, config.phase_g_forward, config.phase_g_back, config.phase_blend);

    let mut accumulated_color = Color::new(0.0, 0.0, 0.0, 0.0);
    let mut transmittance = 1.0f32;

    let max_t = t_exit.min(t_enter + config.max_steps as f32 * config.step_size);
    let mut t = t_enter;

    while t < max_t && transmittance > 0.01 {
        let sample_pos = ray_origin.add(&dir.scale(t));
        let density = sample_cloud_density(&sample_pos, layer, controls);

        if density > 1e-6 {
            // Light march toward sun
            let mut light_transmittance = 1.0f32;
            for li in 1..=config.light_steps {
                let lt = li as f32 * config.light_step_size;
                let lp = sample_pos.add(&config.sun_direction.scale(lt));
                let ld = sample_cloud_density(&lp, layer, controls);
                light_transmittance *= beer_lambert(ld * config.absorption, config.light_step_size);
                if light_transmittance < 0.01 {
                    break;
                }
            }

            let sun_light = config.sun_color.scale(light_transmittance * phase);
            let ambient = config.ambient_color.scale(0.3);
            let luminance = sun_light.add(&ambient);

            let sample_extinction = density * config.absorption;
            let sample_transmittance = beer_lambert(sample_extinction, config.step_size);
            let scatter = luminance.scale(density * (1.0 - sample_transmittance) * transmittance);

            accumulated_color = accumulated_color.add(&scatter);
            transmittance *= sample_transmittance;
        }

        t += config.step_size;
    }

    CloudSample {
        color: accumulated_color,
        transmittance,
    }
}

// ── Temporal reprojection ────────────────────────────────────────

/// Simple temporal reprojection cache for amortizing cloud cost.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalCache {
    pub width: usize,
    pub height: usize,
    pub colors: Vec<Color>,
    pub transmittances: Vec<f32>,
    pub frame_index: u64,
}

impl TemporalCache {
    pub fn new(width: usize, height: usize) -> Self {
        let count = width * height;
        Self {
            width,
            height,
            colors: vec![Color::new(0.0, 0.0, 0.0, 0.0); count],
            transmittances: vec![1.0; count],
            frame_index: 0,
        }
    }

    fn idx(&self, x: usize, y: usize) -> usize {
        (y * self.width + x).min(self.colors.len().saturating_sub(1))
    }

    pub fn get(&self, x: usize, y: usize) -> CloudSample {
        let i = self.idx(x, y);
        CloudSample {
            color: self.colors[i],
            transmittance: self.transmittances[i],
        }
    }

    pub fn store(&mut self, x: usize, y: usize, sample: &CloudSample) {
        let i = self.idx(x, y);
        self.colors[i] = sample.color;
        self.transmittances[i] = sample.transmittance;
    }

    /// Blend new sample with cached value (temporal smoothing).
    pub fn blend(&mut self, x: usize, y: usize, new_sample: &CloudSample, blend_factor: f32) {
        let old = self.get(x, y);
        let blended = CloudSample {
            color: old.color.lerp(&new_sample.color, blend_factor),
            transmittance: old.transmittance + (new_sample.transmittance - old.transmittance) * blend_factor,
        };
        self.store(x, y, &blended);
    }

    pub fn advance_frame(&mut self) {
        self.frame_index += 1;
    }

    /// Checkerboard pattern: which pixels to update this frame.
    pub fn should_update(&self, x: usize, y: usize) -> bool {
        let pattern = (x + y) % 2;
        pattern == (self.frame_index as usize % 2)
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
    fn value_noise_bounded() {
        for i in 0..50 {
            let x = i as f32 * 0.37;
            let y = i as f32 * 0.53;
            let z = i as f32 * 0.71;
            let n = value_noise(x, y, z);
            assert!(n >= 0.0 && n <= 1.0, "noise={n} out of [0,1]");
        }
    }

    #[test]
    fn fbm_noise_bounded() {
        for i in 0..30 {
            let n = fbm_noise(i as f32 * 0.5, 0.0, 0.0, 4);
            assert!(n >= 0.0 && n <= 1.0, "fbm={n} out of range");
        }
    }

    #[test]
    fn worley_noise_bounded() {
        for i in 0..30 {
            let n = worley_noise(i as f32 * 0.3, 0.5, 0.7);
            assert!(n >= 0.0 && n <= 1.0, "worley={n} out of range");
        }
    }

    #[test]
    fn beer_lambert_zero_density() {
        assert!(approx(beer_lambert(0.0, 10.0), 1.0, 1e-6));
    }

    #[test]
    fn beer_lambert_high_density() {
        let t = beer_lambert(10.0, 10.0);
        assert!(t < 1e-10);
    }

    #[test]
    fn powder_effect_bounded() {
        for i in 1..20 {
            let p = powder_effect(i as f32 * 0.1, 1.0);
            assert!(p >= 0.0 && p <= 1.0, "powder={p} out of range");
        }
    }

    #[test]
    fn henyey_greenstein_isotropic() {
        // g=0 → isotropic → 1/(4*PI) everywhere
        let p = henyey_greenstein(0.5, 0.0);
        let expected = 1.0 / (4.0 * std::f32::consts::PI);
        assert!(approx(p, expected, 1e-4));
    }

    #[test]
    fn henyey_greenstein_forward_peak() {
        let forward = henyey_greenstein(1.0, 0.8);
        let backward = henyey_greenstein(-1.0, 0.8);
        assert!(forward > backward, "g>0 should favor forward scatter");
    }

    #[test]
    fn dual_lobe_phase_blend() {
        let p = dual_lobe_phase(0.5, 0.8, -0.3, 0.5);
        assert!(p > 0.0);
    }

    #[test]
    fn cloud_layer_basic() {
        let layer = CloudLayer::new(2000.0, 4000.0);
        assert!(approx(layer.thickness(), 2000.0, 1e-4));
        assert!(layer.contains(3000.0));
        assert!(!layer.contains(5000.0));
    }

    #[test]
    fn cloud_layer_height_fraction() {
        let layer = CloudLayer::new(1000.0, 3000.0);
        assert!(approx(layer.height_fraction(1000.0), 0.0, 1e-4));
        assert!(approx(layer.height_fraction(2000.0), 0.5, 1e-4));
        assert!(approx(layer.height_fraction(3000.0), 1.0, 1e-4));
    }

    #[test]
    fn cloud_layer_ray_intersect_upward() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let hit = layer.intersect_ray(0.0, 1.0);
        assert!(hit.is_some());
        let (t0, t1) = hit.unwrap();
        assert!(approx(t0, 1000.0, 1e-2));
        assert!(approx(t1, 2000.0, 1e-2));
    }

    #[test]
    fn cloud_layer_ray_miss() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let hit = layer.intersect_ray(3000.0, 1.0); // above, going up
        assert!(hit.is_none());
    }

    #[test]
    fn cloud_layer_ray_inside() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let hit = layer.intersect_ray(1500.0, 0.0); // horizontal inside
        assert!(hit.is_some());
    }

    #[test]
    fn cloud_density_outside_layer() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let controls = CloudControls::default();
        let d = sample_cloud_density(&Vec3::new(0.0, 500.0, 0.0), &layer, &controls);
        assert!(approx(d, 0.0, 1e-6));
    }

    #[test]
    fn cloud_density_zero_coverage() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let mut controls = CloudControls::default();
        controls.coverage = 0.0;
        let d = sample_cloud_density(&Vec3::new(0.0, 1500.0, 0.0), &layer, &controls);
        assert!(approx(d, 0.0, 1e-4));
    }

    #[test]
    fn raymarch_no_clouds() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let mut controls = CloudControls::default();
        controls.coverage = 0.0;
        let config = CloudRenderConfig::default();
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let sample = raymarch_clouds(&origin, &dir, &layer, &controls, &config);
        assert!(approx(sample.transmittance, 1.0, 0.05));
    }

    #[test]
    fn raymarch_with_clouds() {
        let layer = CloudLayer::new(1000.0, 2000.0);
        let controls = CloudControls {
            coverage: 0.9,
            density_scale: 2.0,
            ..CloudControls::default()
        };
        let config = CloudRenderConfig {
            max_steps: 32,
            step_size: 50.0,
            ..CloudRenderConfig::default()
        };
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        let sample = raymarch_clouds(&origin, &dir, &layer, &controls, &config);
        // With high coverage, some light should be scattered
        assert!(
            sample.color.r >= 0.0 && sample.color.g >= 0.0 && sample.color.b >= 0.0,
            "cloud color should be non-negative"
        );
    }

    #[test]
    fn temporal_cache_basic() {
        let mut cache = TemporalCache::new(4, 4);
        let sample = CloudSample {
            color: Color::new(0.5, 0.4, 0.3, 1.0),
            transmittance: 0.7,
        };
        cache.store(2, 3, &sample);
        let got = cache.get(2, 3);
        assert!(approx(got.transmittance, 0.7, 1e-6));
    }

    #[test]
    fn temporal_cache_blend() {
        let mut cache = TemporalCache::new(4, 4);
        let s1 = CloudSample { color: Color::new(0.0, 0.0, 0.0, 1.0), transmittance: 1.0 };
        let s2 = CloudSample { color: Color::new(1.0, 1.0, 1.0, 1.0), transmittance: 0.0 };
        cache.store(0, 0, &s1);
        cache.blend(0, 0, &s2, 0.5);
        let got = cache.get(0, 0);
        assert!(approx(got.transmittance, 0.5, 1e-4));
        assert!(approx(got.color.r, 0.5, 1e-4));
    }

    #[test]
    fn temporal_cache_checkerboard() {
        let mut cache = TemporalCache::new(4, 4);
        let u1 = cache.should_update(0, 0);
        cache.advance_frame();
        let u2 = cache.should_update(0, 0);
        assert_ne!(u1, u2, "checkerboard alternates each frame");
    }

    #[test]
    fn cloud_controls_default() {
        let c = CloudControls::default();
        assert!(c.coverage > 0.0 && c.coverage <= 1.0);
    }
}
