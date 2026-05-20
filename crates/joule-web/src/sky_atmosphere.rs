//! Sky and atmosphere rendering: Rayleigh + Mie scattering, sun disc,
//! precomputed transmittance LUT, time-of-day, aerial perspective, night sky,
//! and HDR exposure adaptation.
//!
//! Pure Rust — all atmospheric physics on CPU.

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
    pub fn scale(&self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
    pub fn add(&self, o: &Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
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
    pub fn clamp01(&self) -> Color {
        Color::new(
            self.r.clamp(0.0, 1.0),
            self.g.clamp(0.0, 1.0),
            self.b.clamp(0.0, 1.0),
            self.a.clamp(0.0, 1.0),
        )
    }
}

// ── Physical constants ───────────────────────────────────────────

/// Rayleigh scattering coefficients for RGB (at sea level, per unit length).
const RAYLEIGH_BETA: [f32; 3] = [5.8e-3, 13.5e-3, 33.1e-3];

/// Mie scattering coefficient (wavelength-independent approximation).
const MIE_BETA: f32 = 21.0e-3;

/// Scale height for Rayleigh scattering (km).
const RAYLEIGH_SCALE_HEIGHT: f32 = 8.5;

/// Scale height for Mie scattering (km).
const MIE_SCALE_HEIGHT: f32 = 1.2;

/// Earth radius (km).
const EARTH_RADIUS: f32 = 6371.0;

/// Atmosphere top (km above sea level).
const ATMOSPHERE_HEIGHT: f32 = 80.0;

// ── Rayleigh / Mie ──────────────────────────────────────────────

/// Rayleigh phase function.
pub fn rayleigh_phase(cos_theta: f32) -> f32 {
    0.75 * (1.0 + cos_theta * cos_theta)
}

/// Mie phase function (Henyey-Greenstein with g=0.76).
pub fn mie_phase(cos_theta: f32, asymmetry: f32) -> f32 {
    let g = asymmetry;
    let g2 = g * g;
    let denom = (1.0 + g2 - 2.0 * g * cos_theta).max(1e-10);
    let denom_sqrt = denom.sqrt();
    (1.0 - g2) / (4.0 * std::f32::consts::PI * denom * denom_sqrt)
}

/// Optical depth along a vertical column of atmosphere.
fn optical_depth_vertical(altitude_km: f32, scale_height: f32) -> f32 {
    scale_height * (-altitude_km / scale_height).exp()
}

// ── Sun position ─────────────────────────────────────────────────

/// Compute sun direction from time of day [0, 24).
pub fn sun_direction(hour: f32) -> Vec3 {
    let angle = (hour / 24.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
    Vec3::new(angle.cos(), angle.sin(), 0.3).normalize()
}

/// Sun altitude angle above horizon in radians, from time of day.
pub fn sun_altitude(hour: f32) -> f32 {
    let dir = sun_direction(hour);
    dir.y.asin()
}

/// Sun disc brightness (1 inside disc, soft falloff at edge).
pub fn sun_disc(view_dir: &Vec3, sun_dir: &Vec3, angular_radius: f32) -> f32 {
    let cos_angle = Vec3::dot(view_dir, sun_dir).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    if angle < angular_radius * 0.8 {
        1.0
    } else if angle < angular_radius {
        let t = (angle - angular_radius * 0.8) / (angular_radius * 0.2);
        1.0 - t.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ── Transmittance LUT ────────────────────────────────────────────

/// Precomputed transmittance lookup table.
/// Indexed by (altitude_index, zenith_angle_index).
#[derive(Debug, Clone, PartialEq)]
pub struct TransmittanceLut {
    pub altitude_steps: usize,
    pub zenith_steps: usize,
    pub max_altitude: f32,
    /// Row-major: [alt][zen] * 3 (RGB).
    pub data: Vec<f32>,
}

impl TransmittanceLut {
    pub fn new(altitude_steps: usize, zenith_steps: usize, max_altitude: f32) -> Self {
        let count = altitude_steps * zenith_steps * 3;
        Self {
            altitude_steps,
            zenith_steps,
            max_altitude,
            data: vec![0.0; count],
        }
    }

    fn idx(&self, ai: usize, zi: usize) -> usize {
        (ai * self.zenith_steps + zi) * 3
    }

    pub fn set(&mut self, ai: usize, zi: usize, r: f32, g: f32, b: f32) {
        let i = self.idx(ai, zi);
        if i + 2 < self.data.len() {
            self.data[i] = r;
            self.data[i + 1] = g;
            self.data[i + 2] = b;
        }
    }

    pub fn get(&self, ai: usize, zi: usize) -> (f32, f32, f32) {
        let i = self.idx(ai, zi);
        if i + 2 < self.data.len() {
            (self.data[i], self.data[i + 1], self.data[i + 2])
        } else {
            (0.0, 0.0, 0.0)
        }
    }

    /// Bilinear lookup given altitude (km) and zenith cosine [-1, 1].
    pub fn sample(&self, altitude_km: f32, cos_zenith: f32) -> (f32, f32, f32) {
        let af = (altitude_km / self.max_altitude.max(0.01))
            .clamp(0.0, 1.0)
            * (self.altitude_steps as f32 - 1.0);
        let zf = ((cos_zenith + 1.0) * 0.5).clamp(0.0, 1.0)
            * (self.zenith_steps as f32 - 1.0);
        let ai0 = af.floor() as usize;
        let ai1 = (ai0 + 1).min(self.altitude_steps.saturating_sub(1));
        let zi0 = zf.floor() as usize;
        let zi1 = (zi0 + 1).min(self.zenith_steps.saturating_sub(1));
        let sa = af - af.floor();
        let sz = zf - zf.floor();
        let v00 = self.get(ai0, zi0);
        let v10 = self.get(ai1, zi0);
        let v01 = self.get(ai0, zi1);
        let v11 = self.get(ai1, zi1);
        let lr = |a: f32, b: f32, t: f32| a + (b - a) * t;
        (
            lr(lr(v00.0, v10.0, sa), lr(v01.0, v11.0, sa), sz),
            lr(lr(v00.1, v10.1, sa), lr(v01.1, v11.1, sa), sz),
            lr(lr(v00.2, v10.2, sa), lr(v01.2, v11.2, sa), sz),
        )
    }

    /// Precompute the LUT from scattering parameters.
    pub fn precompute(&mut self) {
        for ai in 0..self.altitude_steps {
            let alt = (ai as f32 / (self.altitude_steps as f32 - 1.0).max(1.0))
                * self.max_altitude;
            for zi in 0..self.zenith_steps {
                let cos_z = (zi as f32 / (self.zenith_steps as f32 - 1.0).max(1.0)) * 2.0 - 1.0;
                // Path length through atmosphere approximation
                let zenith_angle = cos_z.acos();
                let path_scale = if zenith_angle < std::f32::consts::FRAC_PI_2 {
                    1.0 / cos_z.max(0.01)
                } else {
                    // Below horizon: long path
                    10.0
                };
                let rayleigh_od = optical_depth_vertical(alt, RAYLEIGH_SCALE_HEIGHT) * path_scale;
                let mie_od = optical_depth_vertical(alt, MIE_SCALE_HEIGHT) * path_scale;
                let tr = (-RAYLEIGH_BETA[0] * rayleigh_od - MIE_BETA * mie_od).exp();
                let tg = (-RAYLEIGH_BETA[1] * rayleigh_od - MIE_BETA * mie_od).exp();
                let tb = (-RAYLEIGH_BETA[2] * rayleigh_od - MIE_BETA * mie_od).exp();
                self.set(ai, zi, tr, tg, tb);
            }
        }
    }
}

// ── Aerial perspective ───────────────────────────────────────────

/// Fog objects by atmospheric scattering over distance.
pub fn aerial_perspective(
    object_color: &Color,
    fog_color: &Color,
    distance_km: f32,
    density: f32,
) -> Color {
    let t = (-density * distance_km).exp();
    object_color.scale(t).add(&fog_color.scale(1.0 - t)).clamp01()
}

// ── Night sky ────────────────────────────────────────────────────

/// Dark blue gradient for night. Returns sky color at given elevation angle.
pub fn night_sky_color(elevation_rad: f32) -> Color {
    let t = (elevation_rad / std::f32::consts::FRAC_PI_2).clamp(0.0, 1.0);
    let horizon = Color::new(0.01, 0.02, 0.05, 1.0);
    let zenith = Color::new(0.005, 0.01, 0.03, 1.0);
    horizon.lerp(&zenith, t)
}

// ── Exposure / tone-mapping ──────────────────────────────────────

/// Simple Reinhard tone-mapping for HDR sky.
pub fn tonemap_reinhard(color: &Color) -> Color {
    Color::new(
        color.r / (1.0 + color.r),
        color.g / (1.0 + color.g),
        color.b / (1.0 + color.b),
        color.a,
    )
}

/// Exposure-adapted luminance. `exposure` typically 0.1..10.
pub fn apply_exposure(color: &Color, exposure: f32) -> Color {
    color.scale(exposure).clamp01()
}

/// Auto-exposure based on average luminance.
pub fn auto_exposure(avg_luminance: f32, target: f32) -> f32 {
    if avg_luminance < 1e-6 {
        return target;
    }
    target / avg_luminance
}

/// Compute luminance of a color.
pub fn luminance(c: &Color) -> f32 {
    0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b
}

// ── Sky atmosphere compositor ────────────────────────────────────

/// Full atmosphere renderer.
#[derive(Debug, Clone)]
pub struct SkyAtmosphere {
    pub sun_intensity: f32,
    pub sun_color: Color,
    pub sun_angular_radius: f32,
    pub mie_asymmetry: f32,
    pub aerial_density: f32,
    pub lut: TransmittanceLut,
}

impl SkyAtmosphere {
    pub fn new() -> Self {
        let mut lut = TransmittanceLut::new(16, 32, ATMOSPHERE_HEIGHT);
        lut.precompute();
        Self {
            sun_intensity: 20.0,
            sun_color: Color::new(1.0, 0.95, 0.85, 1.0),
            sun_angular_radius: 0.0093, // ~0.53 degrees
            mie_asymmetry: 0.76,
            aerial_density: 0.05,
            lut,
        }
    }

    /// Sample the sky color for a given view direction and sun direction.
    pub fn sample_sky(&self, view_dir: &Vec3, sun_dir: &Vec3) -> Color {
        let view = view_dir.normalize();
        let sun = sun_dir.normalize();
        let cos_theta = Vec3::dot(&view, &sun).clamp(-1.0, 1.0);
        // Elevation-based altitude approximation
        let elevation = view.y.clamp(-0.1, 1.0);
        let altitude_km = elevation.max(0.0) * 10.0; // approximate
        let cos_zenith = view.y;
        let (tr, tg, tb) = self.lut.sample(altitude_km, cos_zenith);
        // Rayleigh inscattering
        let rp = rayleigh_phase(cos_theta);
        let rayleigh = Color::new(
            RAYLEIGH_BETA[0] * rp * tr,
            RAYLEIGH_BETA[1] * rp * tg,
            RAYLEIGH_BETA[2] * rp * tb,
            1.0,
        );
        // Mie inscattering
        let mp = mie_phase(cos_theta, self.mie_asymmetry);
        let mie = Color::new(MIE_BETA * mp * tr, MIE_BETA * mp * tg, MIE_BETA * mp * tb, 1.0);
        let scatter = rayleigh.add(&mie).scale(self.sun_intensity);
        // Sun disc
        let disc = sun_disc(&view, &sun, self.sun_angular_radius);
        let sun_contrib = self.sun_color.scale(disc * self.sun_intensity * 5.0);
        let hdr = scatter.add(&sun_contrib);
        // Night component if sun is below horizon
        if sun.y < 0.05 {
            let night_t = (1.0 - sun.y / 0.05).clamp(0.0, 1.0);
            let night = night_sky_color(view.y.max(0.0));
            return hdr.lerp(&night, night_t);
        }
        hdr
    }

    /// Time-of-day sky color (convenience wrapper).
    pub fn sky_color_at_time(&self, view_dir: &Vec3, hour: f32) -> Color {
        let sun_dir = sun_direction(hour);
        self.sample_sky(view_dir, &sun_dir)
    }

    /// Apply aerial perspective to an object color at given distance.
    pub fn apply_aerial_perspective(
        &self,
        object_color: &Color,
        distance_km: f32,
        sun_dir: &Vec3,
    ) -> Color {
        let fog_color = self.sample_sky(&Vec3::new(0.0, 0.1, 1.0), sun_dir);
        let fog_ldr = tonemap_reinhard(&fog_color);
        aerial_perspective(object_color, &fog_ldr, distance_km, self.aerial_density)
    }
}

impl Default for SkyAtmosphere {
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
    fn rayleigh_phase_forward() {
        let p = rayleigh_phase(1.0);
        assert!(approx(p, 1.5, 1e-4)); // 0.75*(1+1)
    }

    #[test]
    fn rayleigh_phase_perpendicular() {
        let p = rayleigh_phase(0.0);
        assert!(approx(p, 0.75, 1e-4));
    }

    #[test]
    fn mie_phase_forward_peak() {
        let forward = mie_phase(1.0, 0.76);
        let side = mie_phase(0.0, 0.76);
        assert!(forward > side, "Mie should peak forward");
    }

    #[test]
    fn mie_phase_positive() {
        for i in 0..20 {
            let cos_t = -1.0 + i as f32 * 0.1;
            let p = mie_phase(cos_t, 0.76);
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn sun_direction_noon() {
        let d = sun_direction(12.0);
        // At noon (12h), sun should be near the zenith
        assert!(d.y > 0.5, "noon sun should be high: y={}", d.y);
    }

    #[test]
    fn sun_direction_midnight() {
        let d = sun_direction(0.0);
        assert!(d.y < 0.0, "midnight sun should be below horizon");
    }

    #[test]
    fn sun_altitude_daytime() {
        let alt = sun_altitude(12.0);
        assert!(alt > 0.0, "noon altitude should be positive");
    }

    #[test]
    fn sun_disc_inside() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        let view = Vec3::new(0.0, 1.0, 0.0);
        let b = sun_disc(&view, &sun, 0.01);
        assert!(approx(b, 1.0, 1e-4));
    }

    #[test]
    fn sun_disc_outside() {
        let sun = Vec3::new(0.0, 1.0, 0.0);
        let view = Vec3::new(1.0, 0.0, 0.0);
        let b = sun_disc(&view, &sun, 0.01);
        assert!(approx(b, 0.0, 1e-4));
    }

    #[test]
    fn transmittance_lut_precompute() {
        let mut lut = TransmittanceLut::new(8, 16, ATMOSPHERE_HEIGHT);
        lut.precompute();
        let (r, g, b) = lut.get(0, 8); // sea level, ~horizontal
        assert!(r >= 0.0 && g >= 0.0 && b >= 0.0);
    }

    #[test]
    fn transmittance_lut_sample() {
        let mut lut = TransmittanceLut::new(8, 16, ATMOSPHERE_HEIGHT);
        lut.precompute();
        let (r, _g, _b) = lut.sample(0.0, 1.0); // sea level, looking straight up
        assert!(r > 0.0, "transmittance should be positive looking up");
    }

    #[test]
    fn transmittance_lut_altitude_variation() {
        let mut lut = TransmittanceLut::new(8, 16, ATMOSPHERE_HEIGHT);
        lut.precompute();
        let (r_low, _, _) = lut.sample(0.0, 0.5);
        let (r_high, _, _) = lut.sample(40.0, 0.5);
        assert!(r_high >= r_low, "higher altitude should have better transmittance");
    }

    #[test]
    fn aerial_perspective_no_distance() {
        let obj = Color::new(0.5, 0.3, 0.1, 1.0);
        let fog = Color::new(0.5, 0.6, 0.8, 1.0);
        let result = aerial_perspective(&obj, &fog, 0.0, 0.1);
        assert!(approx(result.r, obj.r, 1e-4));
    }

    #[test]
    fn aerial_perspective_large_distance() {
        let obj = Color::new(0.5, 0.3, 0.1, 1.0);
        let fog = Color::new(0.5, 0.6, 0.8, 1.0);
        let result = aerial_perspective(&obj, &fog, 100.0, 0.1);
        // Should converge toward fog
        assert!(
            (result.r - fog.r).abs() < (result.r - obj.r).abs() + 0.1,
            "distant objects approach fog color"
        );
    }

    #[test]
    fn night_sky_dark() {
        let c = night_sky_color(0.5);
        assert!(c.r < 0.1 && c.g < 0.1 && c.b < 0.1);
    }

    #[test]
    fn tonemap_reinhard_clamps() {
        let hdr = Color::new(10.0, 5.0, 1.0, 1.0);
        let ldr = tonemap_reinhard(&hdr);
        assert!(ldr.r < 1.0 && ldr.r > 0.9);
        assert!(ldr.g < 1.0 && ldr.g > 0.7);
    }

    #[test]
    fn exposure_scales() {
        let c = Color::new(0.5, 0.5, 0.5, 1.0);
        let bright = apply_exposure(&c, 2.0);
        assert!(approx(bright.r, 1.0, 1e-6)); // clamped
    }

    #[test]
    fn auto_exposure_adapts() {
        let exp = auto_exposure(0.5, 1.0);
        assert!(approx(exp, 2.0, 1e-4));
    }

    #[test]
    fn luminance_white() {
        let c = Color::new(1.0, 1.0, 1.0, 1.0);
        let l = luminance(&c);
        assert!(approx(l, 1.0, 1e-4));
    }

    #[test]
    fn sky_atmosphere_noon() {
        let atm = SkyAtmosphere::new();
        let view = Vec3::new(0.0, 1.0, 0.0);
        let c = atm.sky_color_at_time(&view, 12.0);
        // Blue sky at noon (blue > red)
        assert!(c.b > c.r, "noon sky should be bluer than red");
    }

    #[test]
    fn sky_atmosphere_night() {
        let atm = SkyAtmosphere::new();
        let view = Vec3::new(0.0, 1.0, 0.0);
        let c = atm.sky_color_at_time(&view, 0.0);
        // Night should be dark
        let ldr = tonemap_reinhard(&c);
        assert!(luminance(&ldr) < 0.3, "night sky should be dark");
    }

    #[test]
    fn sky_atmosphere_aerial() {
        let atm = SkyAtmosphere::new();
        let obj = Color::new(0.3, 0.7, 0.2, 1.0);
        let sun = sun_direction(12.0);
        let result = atm.apply_aerial_perspective(&obj, 5.0, &sun);
        assert!(result.r >= 0.0 && result.r <= 1.0);
    }

    #[test]
    fn sky_atmosphere_default() {
        let atm = SkyAtmosphere::default();
        assert!(atm.sun_intensity > 0.0);
    }
}
