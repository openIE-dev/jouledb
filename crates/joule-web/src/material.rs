//! PBR Material System — albedo, metallic, roughness, emissive, normal maps,
//! Blinn-Phong fallback, material library, and alpha modes.

use std::collections::HashMap;

// ── Color ─────────────────────────────────────────────────────

/// RGBA color with f64 components in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn white() -> Self {
        Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }
    }

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    pub fn lerp(&self, other: &Color, t: f64) -> Color {
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::white()
    }
}

// ── AlphaMode ─────────────────────────────────────────────────

/// How the material handles transparency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlphaMode {
    /// Fully opaque — alpha is ignored.
    Opaque,
    /// Alpha-test: fragments below `cutoff` are discarded.
    Mask { cutoff: f64 },
    /// Alpha blending with the framebuffer.
    Blend,
}

impl Default for AlphaMode {
    fn default() -> Self {
        Self::Opaque
    }
}

// ── TextureRef ────────────────────────────────────────────────

/// UV transform applied to texture coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvTransform {
    pub offset: [f64; 2],
    pub scale: [f64; 2],
    pub rotation: f64,
}

impl Default for UvTransform {
    fn default() -> Self {
        Self {
            offset: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
        }
    }
}

/// A reference to a texture by id, with UV transform.
#[derive(Debug, Clone, PartialEq)]
pub struct TextureRef {
    pub id: String,
    pub uv_transform: UvTransform,
}

impl TextureRef {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.into(),
            uv_transform: UvTransform::default(),
        }
    }

    pub fn with_uv_transform(mut self, uv: UvTransform) -> Self {
        self.uv_transform = uv;
        self
    }
}

// ── PbrMaterial ───────────────────────────────────────────────

/// Physically-Based Rendering material.
#[derive(Debug, Clone, PartialEq)]
pub struct PbrMaterial {
    pub name: String,
    pub albedo: Color,
    pub albedo_map: Option<TextureRef>,
    pub metallic: f64,
    pub roughness: f64,
    pub metallic_roughness_map: Option<TextureRef>,
    pub emissive: Color,
    pub emissive_map: Option<TextureRef>,
    pub normal_map: Option<TextureRef>,
    pub occlusion_map: Option<TextureRef>,
    pub alpha_mode: AlphaMode,
}

impl PbrMaterial {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            albedo: Color::white(),
            albedo_map: None,
            metallic: 0.0,
            roughness: 0.5,
            metallic_roughness_map: None,
            emissive: Color::black(),
            emissive_map: None,
            normal_map: None,
            occlusion_map: None,
            alpha_mode: AlphaMode::Opaque,
        }
    }

    pub fn with_albedo(mut self, color: Color) -> Self {
        self.albedo = color;
        self
    }

    pub fn with_metallic(mut self, metallic: f64) -> Self {
        self.metallic = metallic.clamp(0.0, 1.0);
        self
    }

    pub fn with_roughness(mut self, roughness: f64) -> Self {
        self.roughness = roughness.clamp(0.0, 1.0);
        self
    }

    pub fn with_emissive(mut self, color: Color) -> Self {
        self.emissive = color;
        self
    }

    pub fn with_normal_map(mut self, texture: TextureRef) -> Self {
        self.normal_map = Some(texture);
        self
    }

    pub fn with_alpha_mode(mut self, mode: AlphaMode) -> Self {
        self.alpha_mode = mode;
        self
    }

    /// Is this material transparent (blend or mask)?
    pub fn is_transparent(&self) -> bool {
        !matches!(self.alpha_mode, AlphaMode::Opaque)
    }
}

// ── BlinnPhongMaterial ────────────────────────────────────────

/// Classic Blinn-Phong material as fallback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlinnPhongMaterial {
    pub ambient: Color,
    pub diffuse: Color,
    pub specular: Color,
    pub shininess: f64,
}

impl BlinnPhongMaterial {
    pub fn new() -> Self {
        Self {
            ambient: Color::new(0.1, 0.1, 0.1, 1.0),
            diffuse: Color::new(0.8, 0.8, 0.8, 1.0),
            specular: Color::new(1.0, 1.0, 1.0, 1.0),
            shininess: 32.0,
        }
    }

    /// Convert a PBR material to an approximate Blinn-Phong material.
    pub fn from_pbr(pbr: &PbrMaterial) -> Self {
        let shininess = ((1.0 - pbr.roughness) * 128.0).max(1.0);
        let specular_strength = pbr.metallic * 0.8 + 0.2;
        Self {
            ambient: Color::new(
                pbr.albedo.r * 0.1,
                pbr.albedo.g * 0.1,
                pbr.albedo.b * 0.1,
                pbr.albedo.a,
            ),
            diffuse: pbr.albedo,
            specular: Color::new(specular_strength, specular_strength, specular_strength, 1.0),
            shininess,
        }
    }
}

impl Default for BlinnPhongMaterial {
    fn default() -> Self {
        Self::new()
    }
}

// ── MaterialLibrary ───────────────────────────────────────────

/// A named collection of PBR materials.
pub struct MaterialLibrary {
    materials: HashMap<String, PbrMaterial>,
}

impl MaterialLibrary {
    pub fn new() -> Self {
        Self {
            materials: HashMap::new(),
        }
    }

    pub fn insert(&mut self, mat: PbrMaterial) {
        self.materials.insert(mat.name.clone(), mat);
    }

    pub fn get(&self, name: &str) -> Option<&PbrMaterial> {
        self.materials.get(name)
    }

    pub fn remove(&mut self, name: &str) -> Option<PbrMaterial> {
        self.materials.remove(name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.materials.keys().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }
}

impl Default for MaterialLibrary {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pbr_material() {
        let m = PbrMaterial::new("test");
        assert_eq!(m.albedo, Color::white());
        assert_eq!(m.metallic, 0.0);
        assert_eq!(m.roughness, 0.5);
        assert!(!m.is_transparent());
    }

    #[test]
    fn pbr_builder_chain() {
        let m = PbrMaterial::new("metal")
            .with_albedo(Color::new(0.8, 0.2, 0.1, 1.0))
            .with_metallic(1.0)
            .with_roughness(0.2)
            .with_emissive(Color::new(0.0, 0.5, 0.0, 1.0))
            .with_alpha_mode(AlphaMode::Blend);
        assert_eq!(m.metallic, 1.0);
        assert!(m.is_transparent());
        assert_eq!(m.emissive.g, 0.5);
    }

    #[test]
    fn metallic_clamps() {
        let m = PbrMaterial::new("x").with_metallic(5.0);
        assert_eq!(m.metallic, 1.0);
        let m2 = PbrMaterial::new("x").with_metallic(-1.0);
        assert_eq!(m2.metallic, 0.0);
    }

    #[test]
    fn alpha_mode_mask() {
        let m = PbrMaterial::new("foliage").with_alpha_mode(AlphaMode::Mask { cutoff: 0.5 });
        assert!(m.is_transparent());
        if let AlphaMode::Mask { cutoff } = m.alpha_mode {
            assert_eq!(cutoff, 0.5);
        } else {
            panic!("expected Mask");
        }
    }

    #[test]
    fn blinn_phong_from_pbr() {
        let pbr = PbrMaterial::new("shiny")
            .with_metallic(1.0)
            .with_roughness(0.0);
        let bp = BlinnPhongMaterial::from_pbr(&pbr);
        assert!(bp.shininess >= 127.0);
        assert!(bp.specular.r > 0.9);
    }

    #[test]
    fn material_library_crud() {
        let mut lib = MaterialLibrary::new();
        assert!(lib.is_empty());
        lib.insert(PbrMaterial::new("wood"));
        lib.insert(PbrMaterial::new("stone"));
        assert_eq!(lib.len(), 2);
        assert!(lib.get("wood").is_some());
        lib.remove("wood");
        assert_eq!(lib.len(), 1);
        assert!(lib.get("wood").is_none());
    }

    #[test]
    fn texture_ref_with_uv_transform() {
        let tex = TextureRef::new("diffuse_01").with_uv_transform(UvTransform {
            offset: [0.5, 0.0],
            scale: [2.0, 2.0],
            rotation: std::f64::consts::FRAC_PI_4,
        });
        assert_eq!(tex.uv_transform.scale, [2.0, 2.0]);
    }

    #[test]
    fn color_lerp() {
        let a = Color::black();
        let b = Color::white();
        let mid = a.lerp(&b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-10);
        assert!((mid.g - 0.5).abs() < 1e-10);
        assert!((mid.b - 0.5).abs() < 1e-10);
    }
}
