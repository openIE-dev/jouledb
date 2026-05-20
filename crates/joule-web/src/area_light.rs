//! Area light approximation for the lighting engine.
//!
//! Supports rectangular, disc, and spherical emitter shapes. Uses the
//! Most Significant Point (MSP) method for specular representative
//! points, form-factor diffuse approximation, LTC (Linearly Transformed
//! Cosines) lookup table, energy normalization, and horizon clipping.

use std::f64::consts::PI;

// ── Vector types ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };

    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn dot(self, o: Self) -> f64 { self.x * o.x + self.y * o.y + self.z * o.z }
    pub fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { return Self::ZERO; }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
    }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
    pub fn lerp(self, o: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            x: self.x + (o.x - self.x) * t,
            y: self.y + (o.y - self.y) * t,
            z: self.z + (o.z - self.z) * t,
        }
    }
    pub fn reflect(self, normal: Self) -> Self {
        self.sub(normal.scale(2.0 * self.dot(normal)))
    }
}

/// Linear RGB color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl Color {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };

    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn scale(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
    pub fn add(self, o: Self) -> Self { Self { r: self.r + o.r, g: self.g + o.g, b: self.b + o.b } }
}

// ── Area light shape ───────────────────────────────────────────

/// The emitter shape of an area light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AreaShape {
    /// Rectangular emitter: width (along right) × height (along up).
    Rectangle { width: f64, height: f64 },
    /// Circular disc emitter.
    Disc { radius: f64 },
    /// Spherical emitter.
    Sphere { radius: f64 },
}

impl AreaShape {
    /// Surface area of the emitter.
    pub fn area(&self) -> f64 {
        match self {
            AreaShape::Rectangle { width, height } => width * height,
            AreaShape::Disc { radius } => PI * radius * radius,
            AreaShape::Sphere { radius } => 4.0 * PI * radius * radius,
        }
    }
}

// ── LTC lookup table ───────────────────────────────────────────

/// Precomputed LTC (Linearly Transformed Cosines) lookup table.
/// Stores a 2D table indexed by (roughness, cos_theta) where each
/// entry contains a 3×3 inverse transform matrix (stored as 4 coefficients
/// since the LTC matrix has a known structure: a, 0, b, 0, c, 0, d, 0, 1).
#[derive(Debug, Clone, PartialEq)]
pub struct LtcTable {
    /// Resolution of each axis.
    pub size: u32,
    /// Flattened data: size×size entries, each with 4 f64 (a, b, c, d).
    pub data: Vec<[f64; 4]>,
}

impl LtcTable {
    /// Create an identity LTC table (no transform — equivalent to
    /// a perfectly smooth surface). Useful for testing.
    pub fn identity(size: u32) -> Self {
        let entry = [1.0, 0.0, 1.0, 0.0]; // a=1,b=0,c=1,d=0 → identity
        Self { size, data: vec![entry; (size * size) as usize] }
    }

    /// Create a simple approximation table. Real engines would bake
    /// this from offline computation; here we approximate with
    /// a smooth function of roughness and cos_theta.
    pub fn approximate(size: u32) -> Self {
        let mut data = Vec::with_capacity((size * size) as usize);
        for iy in 0..size {
            let cos_theta = iy as f64 / (size - 1).max(1) as f64;
            for ix in 0..size {
                let roughness = ix as f64 / (size - 1).max(1) as f64;
                let a = 1.0 - roughness * 0.5;
                let b = roughness * 0.3 * cos_theta;
                let c = 1.0 - roughness * 0.3;
                let d = roughness * 0.1;
                data.push([a, b, c, d]);
            }
        }
        Self { size, data }
    }

    /// Look up the LTC coefficients for given roughness and cos_theta (both 0..1).
    pub fn lookup(&self, roughness: f64, cos_theta: f64) -> [f64; 4] {
        if self.size == 0 { return [1.0, 0.0, 1.0, 0.0]; }
        let r = (roughness.clamp(0.0, 1.0) * (self.size - 1) as f64) as u32;
        let c = (cos_theta.clamp(0.0, 1.0) * (self.size - 1) as f64) as u32;
        let r = r.min(self.size - 1);
        let c = c.min(self.size - 1);
        self.data[(c * self.size + r) as usize]
    }
}

// ── AreaLight ──────────────────────────────────────────────────

/// An area light with emitter shape, position, orientation, and energy
/// normalization.
#[derive(Debug, Clone, PartialEq)]
pub struct AreaLight {
    /// Center position of the emitter.
    pub position: Vec3,
    /// Emitter normal (direction it faces).
    pub normal: Vec3,
    /// Local "right" axis (for rectangles / orientation).
    pub right: Vec3,
    /// Local "up" axis.
    pub up: Vec3,
    /// Emitter shape.
    pub shape: AreaShape,
    /// Light color.
    pub color: Color,
    /// Intensity (luminous power).
    pub intensity: f64,
    /// Two-sided emission.
    pub two_sided: bool,
}

impl AreaLight {
    pub fn new(position: Vec3, normal: Vec3, shape: AreaShape, color: Color, intensity: f64) -> Self {
        let n = normal.normalized();
        let right = if n.cross(Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            n.cross(Vec3::new(0.0, 1.0, 0.0)).normalized()
        };
        let up = right.cross(n).normalized();
        Self {
            position,
            normal: n,
            right,
            up,
            shape,
            color,
            intensity,
            two_sided: false,
        }
    }

    pub fn with_two_sided(mut self, two_sided: bool) -> Self {
        self.two_sided = two_sided;
        self
    }

    /// Energy normalization factor. Area lights emit more total energy
    /// proportional to their surface area. This returns the multiplier
    /// so that "intensity" means radiant intensity per solid angle.
    pub fn energy_normalization(&self) -> f64 {
        let area = self.shape.area();
        if area < 1e-12 { return 1.0; }
        // Normalize so that a 1×1 rect has factor 1.
        1.0 / area
    }

    /// Compute the closest point on the emitter to a given world point.
    /// Used for MSP (Most Significant Point) for specular.
    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        match self.shape {
            AreaShape::Rectangle { width, height } => {
                let local = p.sub(self.position);
                let proj_r = local.dot(self.right).clamp(-width * 0.5, width * 0.5);
                let proj_u = local.dot(self.up).clamp(-height * 0.5, height * 0.5);
                self.position
                    .add(self.right.scale(proj_r))
                    .add(self.up.scale(proj_u))
            }
            AreaShape::Disc { radius } => {
                let local = p.sub(self.position);
                let proj_r = local.dot(self.right);
                let proj_u = local.dot(self.up);
                let dist = (proj_r * proj_r + proj_u * proj_u).sqrt();
                if dist < 1e-12 {
                    self.position
                } else {
                    let scale = (dist.min(radius)) / dist;
                    self.position
                        .add(self.right.scale(proj_r * scale))
                        .add(self.up.scale(proj_u * scale))
                }
            }
            AreaShape::Sphere { radius } => {
                let dir = p.sub(self.position);
                let d = dir.length();
                if d < 1e-12 {
                    // Point is at center; return any surface point.
                    self.position.add(self.normal.scale(radius))
                } else {
                    self.position.add(dir.scale(radius / d))
                }
            }
        }
    }

    /// Most Significant Point (MSP) for specular: the point on the
    /// emitter surface that contributes most to the specular highlight
    /// when viewed from `surface_pos` with reflection vector `reflect_dir`.
    pub fn representative_point(
        &self,
        surface_pos: Vec3,
        reflect_dir: Vec3,
    ) -> Vec3 {
        // Project the reflection ray onto the emitter plane and find
        // the closest point on the emitter shape.
        let to_light = self.position.sub(surface_pos);
        let d = to_light.dot(self.normal);
        let denom = reflect_dir.dot(self.normal);

        if denom.abs() < 1e-12 {
            // Reflection is parallel to emitter; use closest point.
            return self.closest_point(surface_pos);
        }

        let t = d / denom;
        if t < 0.0 {
            return self.closest_point(surface_pos);
        }

        let hit = surface_pos.add(reflect_dir.scale(t));
        self.closest_point(hit)
    }

    /// Approximate diffuse contribution using form factor.
    /// Returns the form-factor–weighted color at a surface point with
    /// given normal.
    pub fn diffuse_form_factor(&self, surface_pos: Vec3, surface_normal: Vec3) -> Color {
        let to_light = self.position.sub(surface_pos);
        let dist_sq = to_light.length_sq();
        if dist_sq < 1e-12 { return Color::BLACK; }

        let dist = dist_sq.sqrt();
        let dir = to_light.scale(1.0 / dist);

        // Surface faces light?
        let n_dot_l = surface_normal.normalized().dot(dir).max(0.0);
        if n_dot_l < 1e-12 { return Color::BLACK; }

        // Emitter faces surface? (for one-sided lights)
        let emitter_facing = self.normal.neg().dot(dir);
        if !self.two_sided && emitter_facing < 0.0 { return Color::BLACK; }
        let emitter_cos = if self.two_sided { emitter_facing.abs() } else { emitter_facing.max(0.0) };

        let area = self.shape.area();
        // Approximate form factor: A * cos_surface * cos_emitter / (pi * d^2)
        let form_factor = area * n_dot_l * emitter_cos / (PI * dist_sq);

        self.color.scale(self.intensity * form_factor)
    }

    /// Evaluate diffuse using an LTC table lookup.
    pub fn diffuse_ltc(
        &self,
        surface_pos: Vec3,
        surface_normal: Vec3,
        roughness: f64,
        table: &LtcTable,
    ) -> Color {
        let to_light = self.position.sub(surface_pos);
        let dist = to_light.length();
        if dist < 1e-12 { return Color::BLACK; }

        let dir = to_light.scale(1.0 / dist);
        let cos_theta = surface_normal.normalized().dot(dir).max(0.0);

        let coeffs = table.lookup(roughness, cos_theta);
        // Apply LTC transform to the form factor. The coefficients
        // scale/shear the cosine distribution. Simplified: multiply
        // the form factor by the trace of the LTC matrix.
        let ltc_scale = (coeffs[0] * coeffs[2] - coeffs[1] * coeffs[3]).abs();
        let base = self.diffuse_form_factor(surface_pos, surface_normal);
        Color::new(
            base.r * ltc_scale,
            base.g * ltc_scale,
            base.b * ltc_scale,
        )
    }

    /// Horizon clipping: clip contribution when the emitter is partially
    /// below the surface tangent plane. Returns a multiplier in [0, 1].
    pub fn horizon_clip(&self, surface_pos: Vec3, surface_normal: Vec3) -> f64 {
        // Check how much of the emitter is above the tangent plane.
        let n = surface_normal.normalized();

        match self.shape {
            AreaShape::Rectangle { width, height } => {
                // Test 4 corners.
                let hw = width * 0.5;
                let hh = height * 0.5;
                let corners = [
                    self.position.add(self.right.scale(hw)).add(self.up.scale(hh)),
                    self.position.add(self.right.scale(-hw)).add(self.up.scale(hh)),
                    self.position.add(self.right.scale(hw)).add(self.up.scale(-hh)),
                    self.position.add(self.right.scale(-hw)).add(self.up.scale(-hh)),
                ];
                let mut above = 0u32;
                for c in &corners {
                    let to_c = c.sub(surface_pos);
                    if to_c.dot(n) > 0.0 { above += 1; }
                }
                above as f64 / 4.0
            }
            AreaShape::Disc { radius } | AreaShape::Sphere { radius } => {
                let to_center = self.position.sub(surface_pos);
                let h = to_center.dot(n);
                // Fraction of sphere/disc above the tangent plane.
                if h >= radius { return 1.0; }
                if h <= -radius { return 0.0; }
                (h + radius) / (2.0 * radius)
            }
        }
    }

    /// Combined diffuse contribution with horizon clipping.
    pub fn diffuse_clipped(&self, surface_pos: Vec3, surface_normal: Vec3) -> Color {
        let clip = self.horizon_clip(surface_pos, surface_normal);
        if clip < 1e-12 { return Color::BLACK; }
        let base = self.diffuse_form_factor(surface_pos, surface_normal);
        base.scale(clip)
    }

    /// Rectangle corner positions (only valid for Rectangle shape).
    pub fn rectangle_corners(&self) -> Option<[Vec3; 4]> {
        match self.shape {
            AreaShape::Rectangle { width, height } => {
                let hw = width * 0.5;
                let hh = height * 0.5;
                Some([
                    self.position.add(self.right.scale(-hw)).add(self.up.scale(hh)),
                    self.position.add(self.right.scale(hw)).add(self.up.scale(hh)),
                    self.position.add(self.right.scale(hw)).add(self.up.scale(-hh)),
                    self.position.add(self.right.scale(-hw)).add(self.up.scale(-hh)),
                ])
            }
            _ => None,
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    fn make_rect_light() -> AreaLight {
        AreaLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0), // faces down
            AreaShape::Rectangle { width: 2.0, height: 2.0 },
            Color::WHITE,
            10.0,
        )
    }

    #[test]
    fn area_rectangle() {
        let a = AreaShape::Rectangle { width: 3.0, height: 4.0 };
        assert!(approx(a.area(), 12.0));
    }

    #[test]
    fn area_disc() {
        let a = AreaShape::Disc { radius: 1.0 };
        assert!(approx(a.area(), PI));
    }

    #[test]
    fn area_sphere() {
        let a = AreaShape::Sphere { radius: 1.0 };
        assert!(approx(a.area(), 4.0 * PI));
    }

    #[test]
    fn energy_normalization_unit_rect() {
        let l = AreaLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            AreaShape::Rectangle { width: 1.0, height: 1.0 },
            Color::WHITE,
            1.0,
        );
        assert!(approx(l.energy_normalization(), 1.0));
    }

    #[test]
    fn energy_normalization_large_rect() {
        let l = AreaLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            AreaShape::Rectangle { width: 4.0, height: 4.0 },
            Color::WHITE,
            1.0,
        );
        assert!(approx(l.energy_normalization(), 1.0 / 16.0));
    }

    #[test]
    fn closest_point_rect_center() {
        let l = make_rect_light();
        // Point directly below center.
        let cp = l.closest_point(Vec3::new(0.0, 0.0, 0.0));
        assert!(approx(cp.x, 0.0));
        assert!(approx(cp.y, 5.0));
        assert!(approx(cp.z, 0.0));
    }

    #[test]
    fn closest_point_rect_clamps() {
        let l = make_rect_light();
        // Point far to the side.
        let cp = l.closest_point(Vec3::new(100.0, 0.0, 0.0));
        // Clamped to right edge of 2×2 rect.
        assert!(cp.x.abs() <= 1.0 + EPS);
    }

    #[test]
    fn closest_point_disc() {
        let l = AreaLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            AreaShape::Disc { radius: 2.0 },
            Color::WHITE,
            1.0,
        );
        // Point directly below.
        let cp = l.closest_point(Vec3::new(0.0, 0.0, 0.0));
        assert!(approx(cp.y, 5.0));
    }

    #[test]
    fn closest_point_sphere() {
        let l = AreaLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            AreaShape::Sphere { radius: 1.0 },
            Color::WHITE,
            1.0,
        );
        let cp = l.closest_point(Vec3::ZERO);
        let d = cp.sub(l.position).length();
        assert!(approx(d, 1.0)); // on surface
    }

    #[test]
    fn representative_point_on_axis() {
        let l = make_rect_light();
        let rp = l.representative_point(
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert!(approx(rp.y, 5.0));
    }

    #[test]
    fn diffuse_form_factor_facing() {
        let l = make_rect_light();
        let c = l.diffuse_form_factor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(c.r > 0.0);
    }

    #[test]
    fn diffuse_form_factor_facing_away() {
        let l = make_rect_light();
        let c = l.diffuse_form_factor(Vec3::ZERO, Vec3::new(0.0, -1.0, 0.0));
        assert!(approx(c.r, 0.0));
    }

    #[test]
    fn diffuse_form_factor_one_sided_behind() {
        let l = AreaLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0), // faces UP (away from surface below)
            AreaShape::Rectangle { width: 2.0, height: 2.0 },
            Color::WHITE,
            10.0,
        );
        let c = l.diffuse_form_factor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(c.r, 0.0));
    }

    #[test]
    fn diffuse_two_sided() {
        let l = AreaLight::new(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0), // faces UP
            AreaShape::Rectangle { width: 2.0, height: 2.0 },
            Color::WHITE,
            10.0,
        ).with_two_sided(true);
        let c = l.diffuse_form_factor(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(c.r > 0.0);
    }

    #[test]
    fn horizon_clip_fully_above() {
        let l = make_rect_light();
        let clip = l.horizon_clip(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(clip, 1.0));
    }

    #[test]
    fn horizon_clip_fully_below() {
        let l = make_rect_light();
        // Surface faces up, light is far below.
        let clip = l.horizon_clip(Vec3::new(0.0, 100.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(clip, 0.0));
    }

    #[test]
    fn horizon_clip_sphere_partial() {
        let l = AreaLight::new(
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            AreaShape::Sphere { radius: 2.0 },
            Color::WHITE,
            1.0,
        );
        let clip = l.horizon_clip(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        assert!(clip > 0.0 && clip < 1.0);
    }

    #[test]
    fn ltc_identity_table() {
        let t = LtcTable::identity(8);
        let c = t.lookup(0.5, 0.5);
        assert!(approx(c[0], 1.0));
        assert!(approx(c[1], 0.0));
    }

    #[test]
    fn ltc_approximate_table() {
        let t = LtcTable::approximate(16);
        let smooth = t.lookup(0.0, 1.0);
        let rough = t.lookup(1.0, 1.0);
        // Smooth surface has higher 'a' coefficient.
        assert!(smooth[0] > rough[0]);
    }

    #[test]
    fn diffuse_ltc_produces_color() {
        let l = make_rect_light();
        let table = LtcTable::identity(4);
        let c = l.diffuse_ltc(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.5, &table);
        assert!(c.r > 0.0);
    }

    #[test]
    fn rectangle_corners_count() {
        let l = make_rect_light();
        let corners = l.rectangle_corners();
        assert!(corners.is_some());
        assert_eq!(corners.unwrap().len(), 4);
    }

    #[test]
    fn rectangle_corners_none_for_disc() {
        let l = AreaLight::new(
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            AreaShape::Disc { radius: 1.0 },
            Color::WHITE,
            1.0,
        );
        assert!(l.rectangle_corners().is_none());
    }

    #[test]
    fn diffuse_clipped_matches_horizon() {
        let l = make_rect_light();
        let pos = Vec3::ZERO;
        let norm = Vec3::new(0.0, 1.0, 0.0);
        let clipped = l.diffuse_clipped(pos, norm);
        let raw = l.diffuse_form_factor(pos, norm);
        let clip = l.horizon_clip(pos, norm);
        assert!(approx(clipped.r, raw.r * clip));
    }

    #[test]
    fn vec3_reflect() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalized();
        let n = Vec3::new(0.0, 1.0, 0.0);
        let r = v.reflect(n);
        assert!(approx(r.x, v.x));
        assert!(approx(r.y, -v.y));
    }
}
