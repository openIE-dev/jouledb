//! Anisotropic material for brushed metal, hair, silk.
//!
//! Anisotropy parameters (direction as tangent-space vector or angle, strength
//! 0–1), GGX anisotropic NDF (different roughness along tangent vs bitangent),
//! anisotropic Fresnel, Ward anisotropic model as alternative, flowmap-based
//! anisotropy direction for varying surface grain.
//! Pure Rust — no external math or GPU crate dependencies.

use std::fmt;

// ── Inline vector / color types ─────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 { return Self::new(1.0, 0.0); }
        Self { x: self.x / len, y: self.y / len }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn normalize(self) -> Self {
        let len = self.length();
        if len < 1e-10 { return Self::ZERO; }
        Self { x: self.x / len, y: self.y / len, z: self.z / len }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { x: self.x * s, y: self.y * s, z: self.z * s }
    }

    pub fn add(self, o: Self) -> Self {
        Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z }
    }

    pub fn sub(self, o: Self) -> Self {
        Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z }
    }
}

/// RGB color in linear space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color3 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color3 {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };

    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    pub fn scale(self, s: f32) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(self, o: Self) -> Self {
        Self { r: self.r + o.r, g: self.g + o.g, b: self.b + o.b }
    }

    pub fn mul(self, o: Self) -> Self {
        Self { r: self.r * o.r, g: self.g * o.g, b: self.b * o.b }
    }

    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn clamp01(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }
}

impl Default for Color3 {
    fn default() -> Self {
        Self::WHITE
    }
}

// ── Anisotropy parameters ───────────────────────────────────────

/// Direction of anisotropy on the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnisotropyDirection {
    /// Explicit tangent-space 2D direction (will be normalised).
    Vector(Vec2),
    /// Angle in radians (0 = along tangent, pi/2 = along bitangent).
    Angle(f32),
}

impl AnisotropyDirection {
    /// Resolve to a normalised 2D direction.
    pub fn resolve(&self) -> Vec2 {
        match self {
            AnisotropyDirection::Vector(v) => v.normalize(),
            AnisotropyDirection::Angle(a) => Vec2::new(a.cos(), a.sin()),
        }
    }
}

impl Default for AnisotropyDirection {
    fn default() -> Self {
        AnisotropyDirection::Angle(0.0)
    }
}

/// Anisotropic material parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicParams {
    /// Base roughness (isotropic component).
    pub roughness: f32,
    /// Anisotropy strength (0 = isotropic, 1 = fully anisotropic).
    pub anisotropy: f32,
    /// Anisotropy direction.
    pub direction: AnisotropyDirection,
    /// Base color.
    pub color: Color3,
    /// Metallic factor.
    pub metallic: f32,
}

impl AnisotropicParams {
    pub fn new(roughness: f32, anisotropy: f32) -> Self {
        Self {
            roughness: roughness.clamp(0.01, 1.0),
            anisotropy: anisotropy.clamp(0.0, 1.0),
            direction: AnisotropyDirection::default(),
            color: Color3::new(0.8, 0.8, 0.8),
            metallic: 1.0,
        }
    }

    pub fn with_direction(mut self, dir: AnisotropyDirection) -> Self {
        self.direction = dir;
        self
    }

    pub fn with_color(mut self, c: Color3) -> Self {
        self.color = c;
        self
    }

    pub fn with_metallic(mut self, m: f32) -> Self {
        self.metallic = m.clamp(0.0, 1.0);
        self
    }

    /// Compute roughness along tangent and bitangent axes.
    pub fn roughness_tb(&self) -> (f32, f32) {
        let aspect = (1.0 - self.anisotropy * 0.9).sqrt();
        let alpha_t = (self.roughness * self.roughness / aspect).max(0.001);
        let alpha_b = (self.roughness * self.roughness * aspect).max(0.001);
        (alpha_t, alpha_b)
    }

    /// Rotate the tangent frame by the anisotropy direction.
    pub fn rotate_tangent_frame(&self, tangent: Vec3, bitangent: Vec3) -> (Vec3, Vec3) {
        let dir = self.direction.resolve();
        let t_new = tangent.scale(dir.x).add(bitangent.scale(dir.y)).normalize();
        let b_new = tangent.scale(-dir.y).add(bitangent.scale(dir.x)).normalize();
        (t_new, b_new)
    }
}

impl Default for AnisotropicParams {
    fn default() -> Self {
        Self::new(0.3, 0.5)
    }
}

// ── GGX Anisotropic NDF ─────────────────────────────────────────

/// Anisotropic GGX (Trowbridge-Reitz) normal distribution function.
///
/// `alpha_t` / `alpha_b` are roughness along tangent / bitangent.
/// `t_dot_h`, `b_dot_h`, `n_dot_h` are dot products of half-vector
/// with the tangent, bitangent, and normal respectively.
pub fn ggx_aniso_ndf(
    t_dot_h: f32,
    b_dot_h: f32,
    n_dot_h: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let at2 = alpha_t * alpha_t;
    let ab2 = alpha_b * alpha_b;
    let term = (t_dot_h * t_dot_h / at2)
        + (b_dot_h * b_dot_h / ab2)
        + (n_dot_h * n_dot_h);
    let denom = std::f32::consts::PI * alpha_t * alpha_b * term * term;
    if denom.abs() < 1e-10 { return 0.0; }
    1.0 / denom
}

/// Anisotropic Smith-GGX geometry term (single direction).
fn geometry_aniso_ggx(
    t_dot_v: f32,
    b_dot_v: f32,
    n_dot_v: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let lambda = ((alpha_t * t_dot_v).powi(2) + (alpha_b * b_dot_v).powi(2) + n_dot_v * n_dot_v).sqrt();
    if lambda.abs() < 1e-10 { return 0.0; }
    2.0 * n_dot_v / (n_dot_v + lambda)
}

/// Combined anisotropic geometry function (view + light).
pub fn geometry_aniso_smith(
    t_dot_v: f32,
    b_dot_v: f32,
    n_dot_v: f32,
    t_dot_l: f32,
    b_dot_l: f32,
    n_dot_l: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    geometry_aniso_ggx(t_dot_v, b_dot_v, n_dot_v, alpha_t, alpha_b)
        * geometry_aniso_ggx(t_dot_l, b_dot_l, n_dot_l, alpha_t, alpha_b)
}

// ── Anisotropic Fresnel ─────────────────────────────────────────

/// Schlick Fresnel for anisotropic materials.
/// Uses per-channel F0 (typically metal reflectance).
pub fn fresnel_aniso(f0: Color3, v_dot_h: f32) -> Color3 {
    let c = (1.0 - v_dot_h).clamp(0.0, 1.0);
    let c2 = c * c;
    let c5 = c2 * c2 * c;
    Color3::new(
        f0.r + (1.0 - f0.r) * c5,
        f0.g + (1.0 - f0.g) * c5,
        f0.b + (1.0 - f0.b) * c5,
    )
}

// ── Ward anisotropic model ──────────────────────────────────────

/// Ward anisotropic specular model (alternative to GGX).
///
/// Simpler but less physically accurate — good for artistic control.
pub fn ward_aniso(
    t_dot_h: f32,
    b_dot_h: f32,
    n_dot_l: f32,
    n_dot_v: f32,
    n_dot_h: f32,
    alpha_t: f32,
    alpha_b: f32,
) -> f32 {
    let denom_sqrt = (n_dot_l.max(0.0) * n_dot_v.max(0.0)).sqrt();
    if denom_sqrt < 1e-7 { return 0.0; }
    if n_dot_h < 1e-7 { return 0.0; }

    let exponent = -((t_dot_h / alpha_t).powi(2) + (b_dot_h / alpha_b).powi(2))
        / (n_dot_h * n_dot_h);

    let coeff = 1.0 / (4.0 * std::f32::consts::PI * alpha_t * alpha_b * denom_sqrt);
    (coeff * exponent.exp()).max(0.0)
}

// ── Flowmap-based anisotropy ────────────────────────────────────

/// A 2D flowmap storing per-texel anisotropy directions.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowMap {
    pub width: u32,
    pub height: u32,
    data: Vec<Vec2>,
}

impl FlowMap {
    pub fn new(width: u32, height: u32, data: Vec<Vec2>) -> Result<Self, String> {
        let expected = (width as usize) * (height as usize);
        if data.len() != expected {
            return Err(format!("expected {} entries, got {}", expected, data.len()));
        }
        Ok(Self { width, height, data })
    }

    /// Uniform direction flowmap.
    pub fn uniform(width: u32, height: u32, dir: Vec2) -> Self {
        let count = (width as usize) * (height as usize);
        Self { width, height, data: vec![dir.normalize(); count] }
    }

    /// Radial flowmap (directions point outward from centre).
    pub fn radial(width: u32, height: u32) -> Self {
        let count = (width as usize) * (height as usize);
        let mut data = Vec::with_capacity(count);
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        for y in 0..height {
            for x in 0..width {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-6 {
                    data.push(Vec2::new(1.0, 0.0));
                } else {
                    data.push(Vec2::new(dx / len, dy / len));
                }
            }
        }
        Self { width, height, data }
    }

    /// Sample the flowmap at UV coordinates (nearest).
    pub fn sample(&self, u: f32, v: f32) -> Vec2 {
        let u = u.clamp(0.0, 1.0);
        let v = v.clamp(0.0, 1.0);
        let x = (u * (self.width as f32 - 1.0) + 0.5) as u32;
        let y = (v * (self.height as f32 - 1.0) + 0.5) as u32;
        let x = x.min(self.width - 1);
        let y = y.min(self.height - 1);
        self.data[(y * self.width + x) as usize]
    }
}

impl fmt::Display for FlowMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlowMap({}x{})", self.width, self.height)
    }
}

// ── Full anisotropic BRDF evaluation ────────────────────────────

/// Evaluate the anisotropic GGX BRDF for a single light.
///
/// All vectors in world space, normalised. Returns reflected radiance.
pub fn evaluate_aniso_brdf(
    params: &AnisotropicParams,
    normal: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    view: Vec3,
    light: Vec3,
    light_color: Color3,
) -> Color3 {
    let n = normal.normalize();
    let v = view.normalize();
    let l = light.normalize();
    let h = v.add(l).normalize();

    // Rotate tangent frame by anisotropy direction.
    let (t, b) = params.rotate_tangent_frame(tangent.normalize(), bitangent.normalize());

    let n_dot_v = n.dot(v).max(0.0);
    let n_dot_l = n.dot(l).max(0.0);
    let n_dot_h = n.dot(h).max(0.0);
    let v_dot_h = v.dot(h).max(0.0);
    let t_dot_h = t.dot(h);
    let b_dot_h = b.dot(h);
    let t_dot_v = t.dot(v);
    let b_dot_v = b.dot(v);
    let t_dot_l = t.dot(l);
    let b_dot_l = b.dot(l);

    if n_dot_l <= 0.0 {
        return Color3::BLACK;
    }

    let (alpha_t, alpha_b) = params.roughness_tb();

    // NDF
    let d = ggx_aniso_ndf(t_dot_h, b_dot_h, n_dot_h, alpha_t, alpha_b);

    // Geometry
    let g = geometry_aniso_smith(t_dot_v, b_dot_v, n_dot_v, t_dot_l, b_dot_l, n_dot_l, alpha_t, alpha_b);

    // Fresnel
    let f0 = {
        let dielectric = Color3::new(0.04, 0.04, 0.04);
        Color3::new(
            dielectric.r * (1.0 - params.metallic) + params.color.r * params.metallic,
            dielectric.g * (1.0 - params.metallic) + params.color.g * params.metallic,
            dielectric.b * (1.0 - params.metallic) + params.color.b * params.metallic,
        )
    };
    let f = fresnel_aniso(f0, v_dot_h);

    let denom = (4.0 * n_dot_v * n_dot_l).max(1e-4);
    let specular = Color3::new(
        d * g * f.r / denom,
        d * g * f.g / denom,
        d * g * f.b / denom,
    );

    // Diffuse (Lambert).
    let k_d = Color3::new(
        (1.0 - f.r) * (1.0 - params.metallic),
        (1.0 - f.g) * (1.0 - params.metallic),
        (1.0 - f.b) * (1.0 - params.metallic),
    );
    let diffuse = Color3::new(
        k_d.r * params.color.r / std::f32::consts::PI,
        k_d.g * params.color.g / std::f32::consts::PI,
        k_d.b * params.color.b / std::f32::consts::PI,
    );

    let total = diffuse.add(specular);
    Color3::new(
        total.r * light_color.r * n_dot_l,
        total.g * light_color.g * n_dot_l,
        total.b * light_color.b * n_dot_l,
    )
}

/// Evaluate the Ward model for a single light.
pub fn evaluate_ward_brdf(
    params: &AnisotropicParams,
    normal: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    view: Vec3,
    light: Vec3,
    light_color: Color3,
) -> Color3 {
    let n = normal.normalize();
    let v = view.normalize();
    let l = light.normalize();
    let h = v.add(l).normalize();

    let (t, b) = params.rotate_tangent_frame(tangent.normalize(), bitangent.normalize());

    let n_dot_v = n.dot(v).max(0.0);
    let n_dot_l = n.dot(l).max(0.0);
    let n_dot_h = n.dot(h).max(0.0);
    let t_dot_h = t.dot(h);
    let b_dot_h = b.dot(h);

    if n_dot_l <= 0.0 {
        return Color3::BLACK;
    }

    let (alpha_t, alpha_b) = params.roughness_tb();
    let spec = ward_aniso(t_dot_h, b_dot_h, n_dot_l, n_dot_v, n_dot_h, alpha_t, alpha_b);

    Color3::new(
        params.color.r * spec * light_color.r * n_dot_l,
        params.color.g * spec * light_color.g * n_dot_l,
        params.color.b * spec * light_color.b * n_dot_l,
    )
}

// ── AnisotropicMaterial ─────────────────────────────────────────

/// Complete anisotropic material.
#[derive(Debug, Clone, PartialEq)]
pub struct AnisotropicMaterial {
    pub name: String,
    pub params: AnisotropicParams,
    pub flowmap: Option<FlowMap>,
}

impl AnisotropicMaterial {
    pub fn new(name: impl Into<String>, params: AnisotropicParams) -> Self {
        Self { name: name.into(), params, flowmap: None }
    }

    pub fn with_flowmap(mut self, fm: FlowMap) -> Self {
        self.flowmap = Some(fm);
        self
    }

    /// Get the anisotropy direction at a UV coordinate (uses flowmap if present).
    pub fn direction_at(&self, u: f32, v: f32) -> Vec2 {
        if let Some(fm) = &self.flowmap {
            fm.sample(u, v)
        } else {
            self.params.direction.resolve()
        }
    }
}

impl fmt::Display for AnisotropicMaterial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AnisotropicMaterial({}, rough={:.2}, aniso={:.2})",
            self.name, self.params.roughness, self.params.anisotropy,
        )
    }
}

/// Preset: brushed aluminum.
pub fn preset_brushed_aluminum() -> AnisotropicMaterial {
    AnisotropicMaterial::new(
        "brushed_aluminum",
        AnisotropicParams::new(0.3, 0.7)
            .with_color(Color3::new(0.91, 0.92, 0.92))
            .with_metallic(1.0)
            .with_direction(AnisotropyDirection::Angle(0.0)),
    )
}

/// Preset: silk fabric.
pub fn preset_silk() -> AnisotropicMaterial {
    AnisotropicMaterial::new(
        "silk",
        AnisotropicParams::new(0.5, 0.6)
            .with_color(Color3::new(0.6, 0.1, 0.1))
            .with_metallic(0.0)
            .with_direction(AnisotropyDirection::Angle(std::f32::consts::FRAC_PI_4)),
    )
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    fn approx_wide(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.05
    }

    #[test]
    fn test_anisotropy_direction_angle() {
        let dir = AnisotropyDirection::Angle(0.0).resolve();
        assert!(approx(dir.x, 1.0));
        assert!(approx(dir.y, 0.0));
    }

    #[test]
    fn test_anisotropy_direction_angle_90() {
        let dir = AnisotropyDirection::Angle(std::f32::consts::FRAC_PI_2).resolve();
        assert!(approx(dir.x, 0.0));
        assert!(approx(dir.y, 1.0));
    }

    #[test]
    fn test_anisotropy_direction_vector() {
        let dir = AnisotropyDirection::Vector(Vec2::new(3.0, 4.0)).resolve();
        assert!(approx(dir.x, 0.6));
        assert!(approx(dir.y, 0.8));
    }

    #[test]
    fn test_roughness_tb_isotropic() {
        let p = AnisotropicParams::new(0.5, 0.0);
        let (at, ab) = p.roughness_tb();
        // With zero anisotropy, both should be similar.
        assert!(approx_wide(at, ab));
    }

    #[test]
    fn test_roughness_tb_anisotropic() {
        let p = AnisotropicParams::new(0.5, 0.8);
        let (at, ab) = p.roughness_tb();
        // One should be larger than the other.
        assert!((at - ab).abs() > 0.01);
    }

    #[test]
    fn test_rotate_tangent_frame_identity() {
        let p = AnisotropicParams::new(0.3, 0.5)
            .with_direction(AnisotropyDirection::Angle(0.0));
        let t = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        let (nt, nb) = p.rotate_tangent_frame(t, b);
        assert!(approx(nt.x, 1.0));
        assert!(approx(nb.z, 1.0));
    }

    #[test]
    fn test_ggx_aniso_ndf_positive() {
        let d = ggx_aniso_ndf(0.0, 0.0, 1.0, 0.2, 0.2);
        assert!(d > 0.0);
    }

    #[test]
    fn test_ggx_aniso_ndf_peak_aligned() {
        // H aligned with N: maximum NDF.
        let peak = ggx_aniso_ndf(0.0, 0.0, 1.0, 0.3, 0.3);
        let off = ggx_aniso_ndf(0.5, 0.5, 0.707, 0.3, 0.3);
        assert!(peak > off);
    }

    #[test]
    fn test_geometry_aniso_smith_positive() {
        let g = geometry_aniso_smith(0.3, 0.3, 0.8, 0.3, 0.3, 0.8, 0.2, 0.2);
        assert!(g > 0.0);
        assert!(g <= 1.0 + 1e-6);
    }

    #[test]
    fn test_fresnel_aniso_normal_incidence() {
        let f0 = Color3::new(0.56, 0.57, 0.58);
        let f = fresnel_aniso(f0, 1.0);
        assert!(approx(f.r, 0.56));
    }

    #[test]
    fn test_fresnel_aniso_grazing() {
        let f0 = Color3::new(0.04, 0.04, 0.04);
        let f = fresnel_aniso(f0, 0.0);
        assert!(approx(f.r, 1.0));
    }

    #[test]
    fn test_ward_positive() {
        let spec = ward_aniso(0.1, 0.1, 0.8, 0.8, 0.95, 0.2, 0.2);
        assert!(spec >= 0.0);
    }

    #[test]
    fn test_evaluate_aniso_brdf_head_on() {
        let params = AnisotropicParams::new(0.3, 0.5)
            .with_color(Color3::new(0.9, 0.9, 0.9))
            .with_metallic(1.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        let t = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.0, 1.0, 0.0);
        let result = evaluate_aniso_brdf(&params, n, t, b, v, l, Color3::WHITE);
        assert!(result.luminance() > 0.0);
    }

    #[test]
    fn test_evaluate_aniso_brdf_back_lit() {
        let params = AnisotropicParams::default();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let t = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.0, -1.0, 0.0);
        let result = evaluate_aniso_brdf(&params, n, t, b, v, l, Color3::WHITE);
        assert!(approx(result.r, 0.0));
    }

    #[test]
    fn test_evaluate_ward_brdf_positive() {
        let params = AnisotropicParams::new(0.3, 0.5)
            .with_color(Color3::new(0.5, 0.5, 0.5));
        let n = Vec3::new(0.0, 1.0, 0.0);
        let t = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 0.0, 1.0);
        let v = Vec3::new(0.1, 1.0, 0.0).normalize();
        let l = Vec3::new(0.1, 1.0, 0.0).normalize();
        let result = evaluate_ward_brdf(&params, n, t, b, v, l, Color3::WHITE);
        assert!(result.luminance() >= 0.0);
    }

    #[test]
    fn test_flowmap_uniform() {
        let fm = FlowMap::uniform(4, 4, Vec2::new(1.0, 0.0));
        let d = fm.sample(0.5, 0.5);
        assert!(approx(d.x, 1.0));
        assert!(approx(d.y, 0.0));
    }

    #[test]
    fn test_flowmap_radial() {
        let fm = FlowMap::radial(8, 8);
        // Corner should point away from centre.
        let d = fm.sample(1.0, 1.0);
        assert!(d.x > 0.0);
        assert!(d.y > 0.0);
    }

    #[test]
    fn test_flowmap_creation_error() {
        assert!(FlowMap::new(2, 2, vec![Vec2::new(1.0, 0.0)]).is_err());
    }

    #[test]
    fn test_material_with_flowmap() {
        let fm = FlowMap::uniform(4, 4, Vec2::new(0.0, 1.0));
        let mat = AnisotropicMaterial::new("test", AnisotropicParams::default())
            .with_flowmap(fm);
        let d = mat.direction_at(0.5, 0.5);
        assert!(approx(d.y, 1.0));
    }

    #[test]
    fn test_material_without_flowmap() {
        let mat = AnisotropicMaterial::new(
            "test",
            AnisotropicParams::new(0.3, 0.5).with_direction(AnisotropyDirection::Angle(0.0)),
        );
        let d = mat.direction_at(0.5, 0.5);
        assert!(approx(d.x, 1.0));
    }

    #[test]
    fn test_preset_brushed_aluminum() {
        let m = preset_brushed_aluminum();
        assert_eq!(m.name, "brushed_aluminum");
        assert!(approx(m.params.metallic, 1.0));
        assert!(m.params.anisotropy > 0.5);
    }

    #[test]
    fn test_preset_silk() {
        let m = preset_silk();
        assert!(approx(m.params.metallic, 0.0));
        assert!(m.params.anisotropy > 0.0);
    }

    #[test]
    fn test_display() {
        let m = preset_brushed_aluminum();
        let s = format!("{m}");
        assert!(s.contains("brushed_aluminum"));
        assert!(s.contains("aniso="));
    }

    #[test]
    fn test_flowmap_display() {
        let fm = FlowMap::uniform(8, 8, Vec2::new(1.0, 0.0));
        let s = format!("{fm}");
        assert!(s.contains("8x8"));
    }
}
