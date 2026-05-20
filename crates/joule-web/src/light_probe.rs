//! Light probes for indirect lighting using spherical harmonics.
//!
//! Each probe stores L2 spherical harmonics (9 coefficients per RGB channel
//! = 27 floats). Provides projection of environment onto SH from cubemap
//! samples, SH evaluation for a given normal, probe grid placement with
//! trilinear interpolation between 8 nearest probes, distance-based
//! blending weights, and serialization/deserialization.

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
    pub fn length(self) -> f64 { self.dot(self).sqrt() }
    pub fn length_sq(self) -> f64 { self.dot(self) }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { return Self::ZERO; }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
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
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };

    pub fn new(r: f64, g: f64, b: f64) -> Self { Self { r, g, b } }
    pub fn scale(self, s: f64) -> Self { Self { r: self.r * s, g: self.g * s, b: self.b * s } }
    pub fn add(self, o: Self) -> Self { Self { r: self.r + o.r, g: self.g + o.g, b: self.b + o.b } }
}

// ── Spherical Harmonics (L2) ───────────────────────────────────

/// Number of SH coefficients for L2 (band 0 + band 1 + band 2 = 1+3+5 = 9).
pub const SH_L2_COEFF_COUNT: usize = 9;

/// SH basis functions for a given direction (normalized).
/// Returns 9 basis function values for L2.
pub fn sh_basis_l2(dir: Vec3) -> [f64; SH_L2_COEFF_COUNT] {
    let (x, y, z) = (dir.x, dir.y, dir.z);
    [
        // Band 0
        0.282094791773878,                       // Y_0^0
        // Band 1
        0.488602511902920 * y,                   // Y_1^{-1}
        0.488602511902920 * z,                   // Y_1^0
        0.488602511902920 * x,                   // Y_1^1
        // Band 2
        1.092548430592079 * x * y,               // Y_2^{-2}
        1.092548430592079 * y * z,               // Y_2^{-1}
        0.315391565252520 * (3.0 * z * z - 1.0), // Y_2^0
        1.092548430592079 * x * z,               // Y_2^1
        0.546274215296040 * (x * x - y * y),     // Y_2^2
    ]
}

/// Spherical harmonics coefficients for one color channel (L2, 9 coefficients).
#[derive(Debug, Clone, PartialEq)]
pub struct ShCoeffs {
    pub coeffs: [f64; SH_L2_COEFF_COUNT],
}

impl ShCoeffs {
    pub fn zero() -> Self { Self { coeffs: [0.0; SH_L2_COEFF_COUNT] } }

    /// Evaluate the SH for a given direction.
    pub fn evaluate(&self, dir: Vec3) -> f64 {
        let basis = sh_basis_l2(dir.normalized());
        let mut sum = 0.0;
        for i in 0..SH_L2_COEFF_COUNT {
            sum += self.coeffs[i] * basis[i];
        }
        sum
    }

    /// Add a weighted sample.
    pub fn accumulate(&mut self, dir: Vec3, value: f64, weight: f64) {
        let basis = sh_basis_l2(dir.normalized());
        for i in 0..SH_L2_COEFF_COUNT {
            self.coeffs[i] += value * basis[i] * weight;
        }
    }

    /// Scale all coefficients.
    pub fn scale(&mut self, s: f64) {
        for c in &mut self.coeffs { *c *= s; }
    }

    /// Add another set of coefficients (weighted).
    pub fn add_scaled(&mut self, other: &Self, weight: f64) {
        for i in 0..SH_L2_COEFF_COUNT {
            self.coeffs[i] += other.coeffs[i] * weight;
        }
    }
}

// ── Light Probe ────────────────────────────────────────────────

/// A single light probe storing L2 SH for RGB channels.
#[derive(Debug, Clone, PartialEq)]
pub struct LightProbe {
    /// World-space position.
    pub position: Vec3,
    /// SH coefficients for R, G, B channels.
    pub sh_r: ShCoeffs,
    pub sh_g: ShCoeffs,
    pub sh_b: ShCoeffs,
}

impl LightProbe {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            sh_r: ShCoeffs::zero(),
            sh_g: ShCoeffs::zero(),
            sh_b: ShCoeffs::zero(),
        }
    }

    /// Evaluate the probe to get an irradiance color for a given normal.
    pub fn evaluate(&self, normal: Vec3) -> Color {
        Color::new(
            self.sh_r.evaluate(normal).max(0.0),
            self.sh_g.evaluate(normal).max(0.0),
            self.sh_b.evaluate(normal).max(0.0),
        )
    }

    /// Project a cubemap (6 faces, each face_size × face_size) onto SH.
    /// `faces` is an array of 6 slices of Color, each of length face_size².
    /// Face order: +X, -X, +Y, -Y, +Z, -Z.
    pub fn project_cubemap(&mut self, faces: &[Vec<Color>; 6], face_size: u32) {
        self.sh_r = ShCoeffs::zero();
        self.sh_g = ShCoeffs::zero();
        self.sh_b = ShCoeffs::zero();

        let mut total_weight = 0.0;
        let inv_size = 1.0 / face_size as f64;

        for (face_idx, face) in faces.iter().enumerate() {
            for y in 0..face_size {
                for x in 0..face_size {
                    let u = (x as f64 + 0.5) * inv_size * 2.0 - 1.0;
                    let v = (y as f64 + 0.5) * inv_size * 2.0 - 1.0;

                    let dir = cubemap_direction(face_idx, u, v);
                    let d = dir.length();
                    if d < 1e-12 { continue; }
                    let dir = dir.normalized();

                    // Solid angle weight: accounts for cubemap texel distortion.
                    let solid_angle = 4.0 / (d * d * d * face_size as f64 * face_size as f64);

                    let color = face[(y * face_size + x) as usize];
                    self.sh_r.accumulate(dir, color.r, solid_angle);
                    self.sh_g.accumulate(dir, color.g, solid_angle);
                    self.sh_b.accumulate(dir, color.b, solid_angle);

                    total_weight += solid_angle;
                }
            }
        }

        // Normalize.
        if total_weight > 1e-12 {
            let norm = 4.0 * PI / total_weight;
            self.sh_r.scale(norm);
            self.sh_g.scale(norm);
            self.sh_b.scale(norm);
        }
    }

    /// Serialize the probe to a flat f64 array (position + 27 SH coefficients = 30 floats).
    pub fn serialize(&self) -> Vec<f64> {
        let mut data = Vec::with_capacity(30);
        data.push(self.position.x);
        data.push(self.position.y);
        data.push(self.position.z);
        data.extend_from_slice(&self.sh_r.coeffs);
        data.extend_from_slice(&self.sh_g.coeffs);
        data.extend_from_slice(&self.sh_b.coeffs);
        data
    }

    /// Deserialize from a flat f64 slice. Returns None if wrong length.
    pub fn deserialize(data: &[f64]) -> Option<Self> {
        if data.len() < 30 { return None; }
        let mut probe = Self::new(Vec3::new(data[0], data[1], data[2]));
        probe.sh_r.coeffs.copy_from_slice(&data[3..12]);
        probe.sh_g.coeffs.copy_from_slice(&data[12..21]);
        probe.sh_b.coeffs.copy_from_slice(&data[21..30]);
        Some(probe)
    }
}

/// Map cubemap face + UV to direction vector.
fn cubemap_direction(face: usize, u: f64, v: f64) -> Vec3 {
    match face {
        0 => Vec3::new(1.0, -v, -u),   // +X
        1 => Vec3::new(-1.0, -v, u),   // -X
        2 => Vec3::new(u, 1.0, v),     // +Y
        3 => Vec3::new(u, -1.0, -v),   // -Y
        4 => Vec3::new(u, -v, 1.0),    // +Z
        5 => Vec3::new(-u, -v, -1.0),  // -Z
        _ => Vec3::ZERO,
    }
}

// ── Probe Grid ─────────────────────────────────────────────────

/// AABB bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }
    pub fn extent(&self) -> Vec3 { self.max.sub(self.min) }
}

/// A 3D grid of light probes for spatial interpolation.
#[derive(Debug, Clone)]
pub struct ProbeGrid {
    /// Grid bounds.
    pub bounds: AABB,
    /// Number of probes in each axis.
    pub res_x: u32,
    pub res_y: u32,
    pub res_z: u32,
    /// Probes stored in x-major order: index = z * res_y * res_x + y * res_x + x.
    pub probes: Vec<LightProbe>,
}

impl ProbeGrid {
    /// Create a probe grid with empty (zero-SH) probes.
    pub fn new(bounds: AABB, res_x: u32, res_y: u32, res_z: u32) -> Self {
        let count = (res_x * res_y * res_z) as usize;
        let extent = bounds.extent();
        let dx = if res_x > 1 { extent.x / (res_x - 1) as f64 } else { 0.0 };
        let dy = if res_y > 1 { extent.y / (res_y - 1) as f64 } else { 0.0 };
        let dz = if res_z > 1 { extent.z / (res_z - 1) as f64 } else { 0.0 };

        let mut probes = Vec::with_capacity(count);
        for iz in 0..res_z {
            for iy in 0..res_y {
                for ix in 0..res_x {
                    let pos = Vec3::new(
                        bounds.min.x + ix as f64 * dx,
                        bounds.min.y + iy as f64 * dy,
                        bounds.min.z + iz as f64 * dz,
                    );
                    probes.push(LightProbe::new(pos));
                }
            }
        }
        Self { bounds, res_x, res_y, res_z, probes }
    }

    /// Total number of probes.
    pub fn count(&self) -> usize { self.probes.len() }

    /// Index from 3D coordinates.
    pub fn index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        (iz * self.res_y * self.res_x + iy * self.res_x + ix) as usize
    }

    /// Get a probe by grid coordinates.
    pub fn get(&self, ix: u32, iy: u32, iz: u32) -> Option<&LightProbe> {
        if ix >= self.res_x || iy >= self.res_y || iz >= self.res_z { return None; }
        self.probes.get(self.index(ix, iy, iz))
    }

    /// Get a mutable probe by grid coordinates.
    pub fn get_mut(&mut self, ix: u32, iy: u32, iz: u32) -> Option<&mut LightProbe> {
        if ix >= self.res_x || iy >= self.res_y || iz >= self.res_z { return None; }
        let idx = self.index(ix, iy, iz);
        self.probes.get_mut(idx)
    }

    /// Sample the grid with trilinear interpolation at a world position.
    /// Returns interpolated irradiance color for the given normal.
    pub fn sample(&self, world_pos: Vec3, normal: Vec3) -> Color {
        let extent = self.bounds.extent();

        // Normalized position within the grid.
        let nx = if extent.x > 1e-12 {
            ((world_pos.x - self.bounds.min.x) / extent.x * (self.res_x - 1).max(1) as f64).clamp(0.0, (self.res_x - 1) as f64)
        } else { 0.0 };
        let ny = if extent.y > 1e-12 {
            ((world_pos.y - self.bounds.min.y) / extent.y * (self.res_y - 1).max(1) as f64).clamp(0.0, (self.res_y - 1) as f64)
        } else { 0.0 };
        let nz = if extent.z > 1e-12 {
            ((world_pos.z - self.bounds.min.z) / extent.z * (self.res_z - 1).max(1) as f64).clamp(0.0, (self.res_z - 1) as f64)
        } else { 0.0 };

        let ix0 = (nx.floor() as u32).min(self.res_x.saturating_sub(1));
        let iy0 = (ny.floor() as u32).min(self.res_y.saturating_sub(1));
        let iz0 = (nz.floor() as u32).min(self.res_z.saturating_sub(1));
        let ix1 = (ix0 + 1).min(self.res_x.saturating_sub(1));
        let iy1 = (iy0 + 1).min(self.res_y.saturating_sub(1));
        let iz1 = (iz0 + 1).min(self.res_z.saturating_sub(1));

        let fx = nx - ix0 as f64;
        let fy = ny - iy0 as f64;
        let fz = nz - iz0 as f64;

        // Trilinear interpolation over 8 probes.
        let mut result = Color::BLACK;
        let corners = [
            (ix0, iy0, iz0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
            (ix1, iy0, iz0, fx * (1.0 - fy) * (1.0 - fz)),
            (ix0, iy1, iz0, (1.0 - fx) * fy * (1.0 - fz)),
            (ix1, iy1, iz0, fx * fy * (1.0 - fz)),
            (ix0, iy0, iz1, (1.0 - fx) * (1.0 - fy) * fz),
            (ix1, iy0, iz1, fx * (1.0 - fy) * fz),
            (ix0, iy1, iz1, (1.0 - fx) * fy * fz),
            (ix1, iy1, iz1, fx * fy * fz),
        ];

        for (cx, cy, cz, w) in corners {
            if let Some(probe) = self.get(cx, cy, cz) {
                let c = probe.evaluate(normal);
                result = result.add(c.scale(w));
            }
        }
        result
    }

    /// Compute blending weight for a probe based on distance to a point.
    /// Uses inverse-distance weighting.
    pub fn blend_weight(probe_pos: Vec3, sample_pos: Vec3, falloff: f64) -> f64 {
        let d = probe_pos.sub(sample_pos).length();
        1.0 / (1.0 + d * falloff)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-5;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }
    fn color_approx(a: Color, b: Color) -> bool {
        approx(a.r, b.r) && approx(a.g, b.g) && approx(a.b, b.b)
    }

    #[test]
    fn sh_basis_l2_count() {
        let b = sh_basis_l2(Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(b.len(), 9);
    }

    #[test]
    fn sh_basis_band0_constant() {
        let a = sh_basis_l2(Vec3::new(1.0, 0.0, 0.0));
        let b = sh_basis_l2(Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(a[0], b[0])); // Band 0 is constant.
    }

    #[test]
    fn sh_coeffs_zero_evaluates_zero() {
        let c = ShCoeffs::zero();
        let v = c.evaluate(Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(v, 0.0));
    }

    #[test]
    fn sh_accumulate_and_evaluate() {
        let mut c = ShCoeffs::zero();
        // Accumulate a constant value from all directions.
        let n = 100;
        for i in 0..n {
            let phi = 2.0 * PI * i as f64 / n as f64;
            for j in 0..n {
                let theta = PI * j as f64 / n as f64;
                let dir = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                );
                let weight = 4.0 * PI / (n * n) as f64;
                c.accumulate(dir, 1.0, weight);
            }
        }
        // For a constant environment, band-0 should dominate.
        let v = c.evaluate(Vec3::new(0.0, 0.0, 1.0));
        // Should be approximately 1.0 (constant radiance).
        assert!(v > 0.5, "constant environment should produce positive evaluation, got {}", v);
    }

    #[test]
    fn sh_scale() {
        let mut c = ShCoeffs::zero();
        c.coeffs[0] = 1.0;
        c.scale(3.0);
        assert!(approx(c.coeffs[0], 3.0));
    }

    #[test]
    fn sh_add_scaled() {
        let mut a = ShCoeffs::zero();
        a.coeffs[0] = 1.0;
        let mut b = ShCoeffs::zero();
        b.coeffs[0] = 2.0;
        a.add_scaled(&b, 0.5);
        assert!(approx(a.coeffs[0], 2.0));
    }

    #[test]
    fn probe_new() {
        let p = LightProbe::new(Vec3::new(1.0, 2.0, 3.0));
        assert!(approx(p.position.x, 1.0));
    }

    #[test]
    fn probe_evaluate_zero() {
        let p = LightProbe::new(Vec3::ZERO);
        let c = p.evaluate(Vec3::new(0.0, 0.0, 1.0));
        assert!(color_approx(c, Color::BLACK));
    }

    #[test]
    fn probe_serialize_length() {
        let p = LightProbe::new(Vec3::ZERO);
        let data = p.serialize();
        assert_eq!(data.len(), 30);
    }

    #[test]
    fn probe_serialize_roundtrip() {
        let mut p = LightProbe::new(Vec3::new(1.0, 2.0, 3.0));
        p.sh_r.coeffs[0] = 0.5;
        p.sh_g.coeffs[3] = 1.2;
        p.sh_b.coeffs[8] = -0.3;
        let data = p.serialize();
        let p2 = LightProbe::deserialize(&data).unwrap();
        assert!(approx(p2.position.x, 1.0));
        assert!(approx(p2.sh_r.coeffs[0], 0.5));
        assert!(approx(p2.sh_g.coeffs[3], 1.2));
        assert!(approx(p2.sh_b.coeffs[8], -0.3));
    }

    #[test]
    fn probe_deserialize_short() {
        assert!(LightProbe::deserialize(&[0.0; 5]).is_none());
    }

    #[test]
    fn probe_grid_count() {
        let grid = ProbeGrid::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            3, 3, 3,
        );
        assert_eq!(grid.count(), 27);
    }

    #[test]
    fn probe_grid_positions() {
        let grid = ProbeGrid::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
        );
        let p = grid.get(0, 0, 0).unwrap();
        assert!(approx(p.position.x, 0.0));
        let p = grid.get(1, 1, 1).unwrap();
        assert!(approx(p.position.x, 10.0));
    }

    #[test]
    fn probe_grid_get_out_of_bounds() {
        let grid = ProbeGrid::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
        );
        assert!(grid.get(5, 0, 0).is_none());
    }

    #[test]
    fn probe_grid_sample_at_probe_position() {
        let mut grid = ProbeGrid::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
        );
        // Set a constant SH on all probes (band-0 only).
        let band0_value = 1.0 / 0.282094791773878; // normalize so evaluate gives ~1.0
        for p in &mut grid.probes {
            p.sh_r.coeffs[0] = band0_value;
        }
        let c = grid.sample(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(c.r > 0.0);
    }

    #[test]
    fn probe_grid_sample_clamped_outside() {
        let grid = ProbeGrid::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
        );
        // Sampling outside bounds should clamp, not crash.
        let _c = grid.sample(Vec3::new(-100.0, -100.0, -100.0), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn blend_weight_at_zero_distance() {
        let w = ProbeGrid::blend_weight(Vec3::ZERO, Vec3::ZERO, 1.0);
        assert!(approx(w, 1.0));
    }

    #[test]
    fn blend_weight_decreases_with_distance() {
        let w1 = ProbeGrid::blend_weight(Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), 1.0);
        let w2 = ProbeGrid::blend_weight(Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!(w1 > w2);
    }

    #[test]
    fn cubemap_direction_positive_z() {
        let d = cubemap_direction(4, 0.0, 0.0); // +Z face, center
        assert!(d.z > 0.0);
    }

    #[test]
    fn probe_project_uniform_cubemap() {
        let mut probe = LightProbe::new(Vec3::ZERO);
        let face_size = 4u32;
        let white = vec![Color::WHITE; (face_size * face_size) as usize];
        let faces = [
            white.clone(), white.clone(), white.clone(),
            white.clone(), white.clone(), white.clone(),
        ];
        probe.project_cubemap(&faces, face_size);
        // After projecting uniform white, band-0 should be significant.
        let c = probe.evaluate(Vec3::new(0.0, 0.0, 1.0));
        assert!(c.r > 0.0);
        assert!(c.g > 0.0);
        assert!(c.b > 0.0);
    }

    #[test]
    fn probe_grid_get_mut() {
        let mut grid = ProbeGrid::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
        );
        let p = grid.get_mut(0, 0, 0).unwrap();
        p.sh_r.coeffs[0] = 42.0;
        assert!(approx(grid.get(0, 0, 0).unwrap().sh_r.coeffs[0], 42.0));
    }
}
