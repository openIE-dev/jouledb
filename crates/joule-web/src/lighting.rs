//! Lighting — point, directional, and spot lights with attenuation,
//! Phong shading, ambient light, and shadow map configuration.

use crate::webgl::Vec3;

// ── Light Color ───────────────────────────────────────────────

/// RGB light color (linear, unbounded — values > 1.0 for HDR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl LightColor {
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0 }
    }

    pub fn scale(&self, factor: f64) -> Self {
        Self {
            r: self.r * factor,
            g: self.g * factor,
            b: self.b * factor,
        }
    }

    pub fn add(&self, other: &LightColor) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }

    pub fn multiply(&self, other: &LightColor) -> Self {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }
}

impl Default for LightColor {
    fn default() -> Self {
        Self::white()
    }
}

// ── Attenuation ───────────────────────────────────────────────

/// Distance attenuation factors: 1.0 / (constant + linear*d + quadratic*d^2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attenuation {
    pub constant: f64,
    pub linear: f64,
    pub quadratic: f64,
}

impl Attenuation {
    pub fn new(constant: f64, linear: f64, quadratic: f64) -> Self {
        Self { constant, linear, quadratic }
    }

    /// No attenuation — always full intensity.
    pub fn none() -> Self {
        Self { constant: 1.0, linear: 0.0, quadratic: 0.0 }
    }

    /// Compute attenuation factor at a given distance.
    pub fn factor(&self, distance: f64) -> f64 {
        let denom = self.constant + self.linear * distance + self.quadratic * distance * distance;
        if denom < 1e-12 {
            1.0
        } else {
            1.0 / denom
        }
    }
}

impl Default for Attenuation {
    fn default() -> Self {
        Self::none()
    }
}

// ── Light Types ───────────────────────────────────────────────

/// Ambient light — uniform illumination with no direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientLight {
    pub color: LightColor,
    pub intensity: f64,
}

impl AmbientLight {
    pub fn new(color: LightColor, intensity: f64) -> Self {
        Self { color, intensity }
    }

    /// Contribution of ambient light to a surface color.
    pub fn contribute(&self, surface_color: &LightColor) -> LightColor {
        self.color.scale(self.intensity).multiply(surface_color)
    }
}

/// Directional light — parallel rays from infinitely far away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: LightColor,
    pub intensity: f64,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: LightColor, intensity: f64) -> Self {
        Self {
            direction: direction.normalize(),
            color,
            intensity,
        }
    }
}

/// Point light — radiates in all directions from a position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointLight {
    pub position: Vec3,
    pub color: LightColor,
    pub intensity: f64,
    pub attenuation: Attenuation,
}

impl PointLight {
    pub fn new(position: Vec3, color: LightColor, intensity: f64) -> Self {
        Self {
            position,
            color,
            intensity,
            attenuation: Attenuation::new(1.0, 0.09, 0.032),
        }
    }

    pub fn with_attenuation(mut self, attenuation: Attenuation) -> Self {
        self.attenuation = attenuation;
        self
    }
}

/// Spot light — cone of light from a position in a direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpotLight {
    pub position: Vec3,
    pub direction: Vec3,
    pub color: LightColor,
    pub intensity: f64,
    pub attenuation: Attenuation,
    /// Inner cutoff angle (radians) — full intensity inside this cone.
    pub inner_cutoff: f64,
    /// Outer cutoff angle (radians) — zero intensity outside this cone.
    pub outer_cutoff: f64,
}

impl SpotLight {
    pub fn new(
        position: Vec3,
        direction: Vec3,
        color: LightColor,
        intensity: f64,
        inner_cutoff: f64,
        outer_cutoff: f64,
    ) -> Self {
        Self {
            position,
            direction: direction.normalize(),
            color,
            intensity,
            attenuation: Attenuation::new(1.0, 0.09, 0.032),
            inner_cutoff,
            outer_cutoff,
        }
    }

    /// Compute the spotlight intensity factor for a given surface position.
    pub fn spot_factor(&self, surface_pos: &Vec3) -> f64 {
        let light_dir = (*surface_pos - self.position).normalize();
        let cos_angle = light_dir.dot(&self.direction);
        let cos_inner = self.inner_cutoff.cos();
        let cos_outer = self.outer_cutoff.cos();
        if cos_angle > cos_inner {
            1.0
        } else if cos_angle > cos_outer {
            let denom = cos_inner - cos_outer;
            if denom.abs() < 1e-12 {
                0.0
            } else {
                (cos_angle - cos_outer) / denom
            }
        } else {
            0.0
        }
    }
}

// ── Shadow Map Config ─────────────────────────────────────────

/// Configuration for shadow mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowMapConfig {
    pub resolution: u32,
    pub bias: f64,
    pub near: f64,
    pub far: f64,
    pub enabled: bool,
}

impl ShadowMapConfig {
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            bias: 0.005,
            near: 0.1,
            far: 100.0,
            enabled: true,
        }
    }
}

impl Default for ShadowMapConfig {
    fn default() -> Self {
        Self::new(1024)
    }
}

// ── Phong Shading ─────────────────────────────────────────────

/// Compute Phong diffuse + specular for a single light contribution.
///
/// - `normal`: surface normal (must be normalized).
/// - `view_dir`: direction from surface toward the camera (normalized).
/// - `light_dir`: direction from surface toward the light (normalized).
/// - `light_color`: color of the light (already scaled by intensity/attenuation).
/// - `surface_color`: base surface color.
/// - `shininess`: specular exponent.
/// - `specular_strength`: specular multiplier (0..1).
pub fn phong_shade(
    normal: &Vec3,
    view_dir: &Vec3,
    light_dir: &Vec3,
    light_color: &LightColor,
    surface_color: &LightColor,
    shininess: f64,
    specular_strength: f64,
) -> LightColor {
    // Diffuse.
    let n_dot_l = normal.dot(light_dir).max(0.0);
    let diffuse = light_color.multiply(surface_color).scale(n_dot_l);

    // Specular (Blinn-Phong: use halfway vector).
    let halfway = (*light_dir + *view_dir).normalize();
    let n_dot_h = normal.dot(&halfway).max(0.0);
    let spec = n_dot_h.powf(shininess) * specular_strength;
    let specular = light_color.scale(spec);

    diffuse.add(&specular)
}

/// Shade a surface point with a directional light + ambient.
pub fn shade_directional(
    normal: &Vec3,
    view_dir: &Vec3,
    light: &DirectionalLight,
    ambient: &AmbientLight,
    surface_color: &LightColor,
    shininess: f64,
) -> LightColor {
    let light_dir = light.direction * -1.0; // toward the light
    let light_col = light.color.scale(light.intensity);
    let direct = phong_shade(normal, view_dir, &light_dir, &light_col, surface_color, shininess, 0.5);
    let amb = ambient.contribute(surface_color);
    amb.add(&direct)
}

/// Shade a surface point with a point light + ambient.
pub fn shade_point_light(
    surface_pos: &Vec3,
    normal: &Vec3,
    view_dir: &Vec3,
    light: &PointLight,
    ambient: &AmbientLight,
    surface_color: &LightColor,
    shininess: f64,
) -> LightColor {
    let to_light = light.position - *surface_pos;
    let distance = to_light.length();
    let light_dir = if distance > 1e-12 { to_light * (1.0 / distance) } else { Vec3::up() };
    let atten = light.attenuation.factor(distance);
    let light_col = light.color.scale(light.intensity * atten);
    let direct = phong_shade(normal, view_dir, &light_dir, &light_col, surface_color, shininess, 0.5);
    let amb = ambient.contribute(surface_color);
    amb.add(&direct)
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_4;

    const EPS: f64 = 1e-6;

    #[test]
    fn attenuation_at_zero_distance() {
        let a = Attenuation::new(1.0, 0.09, 0.032);
        assert!((a.factor(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn attenuation_decreases_with_distance() {
        let a = Attenuation::new(1.0, 0.09, 0.032);
        assert!(a.factor(10.0) < a.factor(1.0));
    }

    #[test]
    fn ambient_contribution() {
        let amb = AmbientLight::new(LightColor::white(), 0.2);
        let surface = LightColor::new(1.0, 0.0, 0.0);
        let result = amb.contribute(&surface);
        assert!((result.r - 0.2).abs() < EPS);
        assert!((result.g).abs() < EPS);
    }

    #[test]
    fn phong_diffuse_facing_light() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let view = Vec3::new(0.0, 1.0, 0.0);
        let light_dir = Vec3::new(0.0, 1.0, 0.0); // directly above
        let lc = LightColor::white();
        let sc = LightColor::white();
        let result = phong_shade(&normal, &view, &light_dir, &lc, &sc, 32.0, 0.5);
        // Full diffuse (1.0) + specular (0.5 * 1^32 = 0.5) = 1.5
        assert!((result.r - 1.5).abs() < EPS);
    }

    #[test]
    fn phong_no_light_from_behind() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let view = Vec3::new(0.0, 1.0, 0.0);
        let light_dir = Vec3::new(0.0, -1.0, 0.0); // from below
        let lc = LightColor::white();
        let sc = LightColor::white();
        let result = phong_shade(&normal, &view, &light_dir, &lc, &sc, 32.0, 0.5);
        // No diffuse, no specular.
        assert!(result.r < EPS);
    }

    #[test]
    fn spot_light_inside_cone() {
        let spot = SpotLight::new(
            Vec3::zero(),
            Vec3::new(0.0, -1.0, 0.0),
            LightColor::white(),
            1.0,
            0.3,
            0.5,
        );
        // Point directly below.
        let surface = Vec3::new(0.0, -5.0, 0.0);
        let factor = spot.spot_factor(&surface);
        assert!((factor - 1.0).abs() < EPS);
    }

    #[test]
    fn spot_light_outside_cone() {
        let spot = SpotLight::new(
            Vec3::zero(),
            Vec3::new(0.0, -1.0, 0.0),
            LightColor::white(),
            1.0,
            0.1,
            0.2,
        );
        // Point to the side, well outside the cone.
        let surface = Vec3::new(10.0, 0.0, 0.0);
        let factor = spot.spot_factor(&surface);
        assert!(factor < EPS);
    }

    #[test]
    fn shadow_map_config_defaults() {
        let c = ShadowMapConfig::default();
        assert_eq!(c.resolution, 1024);
        assert!(c.enabled);
        assert!(c.bias > 0.0);
    }

    #[test]
    fn shade_directional_produces_color() {
        let normal = Vec3::new(0.0, 1.0, 0.0);
        let view = Vec3::new(0.0, 1.0, 0.0);
        let light = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), LightColor::white(), 1.0);
        let ambient = AmbientLight::new(LightColor::white(), 0.1);
        let surface = LightColor::new(0.8, 0.2, 0.1);
        let result = shade_directional(&normal, &view, &light, &ambient, &surface, 32.0);
        // Should have ambient + direct components.
        assert!(result.r > 0.08); // at least ambient
    }

    #[test]
    fn light_color_operations() {
        let a = LightColor::new(0.5, 0.3, 0.1);
        let b = LightColor::new(0.2, 0.4, 0.6);
        let sum = a.add(&b);
        assert!((sum.r - 0.7).abs() < EPS);
        let prod = a.multiply(&b);
        assert!((prod.r - 0.1).abs() < EPS);
        let scaled = a.scale(2.0);
        assert!((scaled.r - 1.0).abs() < EPS);
    }
}
