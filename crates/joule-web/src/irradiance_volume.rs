//! Irradiance volume for GI (Global Illumination) approximation.
//!
//! A 3D grid of irradiance probes storing spherical harmonics (L1 or L2).
//! Provides volume bounds (AABB), configurable resolution, trilinear
//! interpolation sampling, probe validity flags, cascaded volumes
//! (high-res near camera, low-res far), and blending between volumes.

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
    pub fn lerp(self, o: Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: self.r + (o.r - self.r) * t,
            g: self.g + (o.g - self.g) * t,
            b: self.b + (o.b - self.b) * t,
        }
    }
}

// ── Spherical Harmonics ────────────────────────────────────────

/// SH order for irradiance probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShOrder {
    /// L1: 4 coefficients (band 0 + band 1).
    L1,
    /// L2: 9 coefficients (band 0 + band 1 + band 2).
    L2,
}

impl ShOrder {
    pub fn coeff_count(self) -> usize {
        match self {
            ShOrder::L1 => 4,
            ShOrder::L2 => 9,
        }
    }
}

/// SH coefficients for one color channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ShCoeffs {
    pub coeffs: Vec<f64>,
}

impl ShCoeffs {
    pub fn zero(order: ShOrder) -> Self {
        Self { coeffs: vec![0.0; order.coeff_count()] }
    }

    /// Evaluate for a given direction.
    pub fn evaluate(&self, dir: Vec3) -> f64 {
        let d = dir.normalized();
        let (x, y, z) = (d.x, d.y, d.z);
        let n = self.coeffs.len();
        let mut sum = 0.0;

        // Band 0
        if n > 0 { sum += self.coeffs[0] * 0.282094791773878; }
        // Band 1
        if n > 1 { sum += self.coeffs[1] * 0.488602511902920 * y; }
        if n > 2 { sum += self.coeffs[2] * 0.488602511902920 * z; }
        if n > 3 { sum += self.coeffs[3] * 0.488602511902920 * x; }
        // Band 2
        if n > 4 { sum += self.coeffs[4] * 1.092548430592079 * x * y; }
        if n > 5 { sum += self.coeffs[5] * 1.092548430592079 * y * z; }
        if n > 6 { sum += self.coeffs[6] * 0.315391565252520 * (3.0 * z * z - 1.0); }
        if n > 7 { sum += self.coeffs[7] * 1.092548430592079 * x * z; }
        if n > 8 { sum += self.coeffs[8] * 0.546274215296040 * (x * x - y * y); }

        sum
    }

    pub fn add_scaled(&mut self, other: &Self, weight: f64) {
        let n = self.coeffs.len().min(other.coeffs.len());
        for i in 0..n {
            self.coeffs[i] += other.coeffs[i] * weight;
        }
    }

    pub fn scale(&mut self, s: f64) {
        for c in &mut self.coeffs { *c *= s; }
    }
}

// ── AABB ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self { Self { min, max } }

    pub fn extent(&self) -> Vec3 { self.max.sub(self.min) }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        )
    }

    pub fn contains(&self, p: Vec3) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }
}

// ── Irradiance Probe ───────────────────────────────────────────

/// A single irradiance probe.
#[derive(Debug, Clone, PartialEq)]
pub struct IrradianceProbe {
    /// Position in world space.
    pub position: Vec3,
    /// SH coefficients for R, G, B.
    pub sh_r: ShCoeffs,
    pub sh_g: ShCoeffs,
    pub sh_b: ShCoeffs,
    /// Whether this probe is valid (probes inside geometry are invalid).
    pub valid: bool,
}

impl IrradianceProbe {
    pub fn new(position: Vec3, order: ShOrder) -> Self {
        Self {
            position,
            sh_r: ShCoeffs::zero(order),
            sh_g: ShCoeffs::zero(order),
            sh_b: ShCoeffs::zero(order),
            valid: true,
        }
    }

    /// Evaluate irradiance for a given normal direction.
    pub fn evaluate(&self, normal: Vec3) -> Color {
        if !self.valid { return Color::BLACK; }
        Color::new(
            self.sh_r.evaluate(normal).max(0.0),
            self.sh_g.evaluate(normal).max(0.0),
            self.sh_b.evaluate(normal).max(0.0),
        )
    }

    /// Mark this probe as invalid (e.g., inside geometry).
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
}

// ── Irradiance Volume ──────────────────────────────────────────

/// A 3D grid of irradiance probes.
#[derive(Debug, Clone)]
pub struct IrradianceVolume {
    /// World-space bounding box.
    pub bounds: AABB,
    /// Resolution in each axis.
    pub res_x: u32,
    pub res_y: u32,
    pub res_z: u32,
    /// SH order used.
    pub sh_order: ShOrder,
    /// Probes in x-major order.
    pub probes: Vec<IrradianceProbe>,
}

impl IrradianceVolume {
    /// Create a new volume with empty probes.
    pub fn new(bounds: AABB, res_x: u32, res_y: u32, res_z: u32, order: ShOrder) -> Self {
        let count = (res_x * res_y * res_z) as usize;
        let ext = bounds.extent();
        let dx = if res_x > 1 { ext.x / (res_x - 1) as f64 } else { 0.0 };
        let dy = if res_y > 1 { ext.y / (res_y - 1) as f64 } else { 0.0 };
        let dz = if res_z > 1 { ext.z / (res_z - 1) as f64 } else { 0.0 };

        let mut probes = Vec::with_capacity(count);
        for iz in 0..res_z {
            for iy in 0..res_y {
                for ix in 0..res_x {
                    let pos = Vec3::new(
                        bounds.min.x + ix as f64 * dx,
                        bounds.min.y + iy as f64 * dy,
                        bounds.min.z + iz as f64 * dz,
                    );
                    probes.push(IrradianceProbe::new(pos, order));
                }
            }
        }
        Self { bounds, res_x, res_y, res_z, sh_order: order, probes }
    }

    pub fn count(&self) -> usize { self.probes.len() }

    pub fn index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        (iz * self.res_y * self.res_x + iy * self.res_x + ix) as usize
    }

    pub fn get(&self, ix: u32, iy: u32, iz: u32) -> Option<&IrradianceProbe> {
        if ix >= self.res_x || iy >= self.res_y || iz >= self.res_z { return None; }
        self.probes.get(self.index(ix, iy, iz))
    }

    pub fn get_mut(&mut self, ix: u32, iy: u32, iz: u32) -> Option<&mut IrradianceProbe> {
        if ix >= self.res_x || iy >= self.res_y || iz >= self.res_z { return None; }
        let idx = self.index(ix, iy, iz);
        self.probes.get_mut(idx)
    }

    /// Whether a world position is inside this volume's bounds.
    pub fn contains(&self, pos: Vec3) -> bool {
        self.bounds.contains(pos)
    }

    /// Count valid probes.
    pub fn valid_count(&self) -> usize {
        self.probes.iter().filter(|p| p.valid).count()
    }

    /// Count invalid probes.
    pub fn invalid_count(&self) -> usize {
        self.probes.iter().filter(|p| !p.valid).count()
    }

    /// Sample irradiance at a world position with trilinear interpolation.
    /// Invalid probes are skipped; their weight is redistributed.
    pub fn sample(&self, world_pos: Vec3, normal: Vec3) -> Color {
        let ext = self.bounds.extent();

        let nx = if ext.x > 1e-12 {
            ((world_pos.x - self.bounds.min.x) / ext.x * (self.res_x - 1).max(1) as f64)
                .clamp(0.0, (self.res_x - 1) as f64)
        } else { 0.0 };
        let ny = if ext.y > 1e-12 {
            ((world_pos.y - self.bounds.min.y) / ext.y * (self.res_y - 1).max(1) as f64)
                .clamp(0.0, (self.res_y - 1) as f64)
        } else { 0.0 };
        let nz = if ext.z > 1e-12 {
            ((world_pos.z - self.bounds.min.z) / ext.z * (self.res_z - 1).max(1) as f64)
                .clamp(0.0, (self.res_z - 1) as f64)
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

        let mut result = Color::BLACK;
        let mut total_weight = 0.0;

        for (cx, cy, cz, w) in corners {
            if let Some(probe) = self.get(cx, cy, cz) {
                if probe.valid {
                    let c = probe.evaluate(normal);
                    result = result.add(c.scale(w));
                    total_weight += w;
                }
            }
        }

        // Re-normalize if some probes were invalid.
        if total_weight > 1e-12 && total_weight < 1.0 - 1e-6 {
            result = result.scale(1.0 / total_weight);
        }
        result
    }

    /// Probe spacing along each axis.
    pub fn spacing(&self) -> Vec3 {
        let ext = self.bounds.extent();
        Vec3::new(
            if self.res_x > 1 { ext.x / (self.res_x - 1) as f64 } else { ext.x },
            if self.res_y > 1 { ext.y / (self.res_y - 1) as f64 } else { ext.y },
            if self.res_z > 1 { ext.z / (self.res_z - 1) as f64 } else { ext.z },
        )
    }
}

// ── Cascaded Irradiance Volumes ────────────────────────────────

/// A set of irradiance volumes at different resolutions/scales.
/// Volumes earlier in the list are higher resolution (near camera).
#[derive(Debug, Clone)]
pub struct CascadedIrradianceVolumes {
    /// Ordered list of volumes (index 0 = highest res / nearest).
    pub volumes: Vec<IrradianceVolume>,
}

impl CascadedIrradianceVolumes {
    pub fn new() -> Self { Self { volumes: Vec::new() } }

    pub fn add_volume(&mut self, volume: IrradianceVolume) {
        self.volumes.push(volume);
    }

    /// Create cascaded volumes centered at `center` with doubling extents.
    pub fn create_cascaded(
        center: Vec3,
        cascade_count: u32,
        base_extent: f64,
        base_resolution: u32,
        order: ShOrder,
    ) -> Self {
        let mut volumes = Vec::with_capacity(cascade_count as usize);
        for i in 0..cascade_count {
            let scale = (1u32 << i) as f64;
            let half_ext = base_extent * scale * 0.5;
            let bounds = AABB::new(
                Vec3::new(center.x - half_ext, center.y - half_ext, center.z - half_ext),
                Vec3::new(center.x + half_ext, center.y + half_ext, center.z + half_ext),
            );
            // Resolution stays constant or decreases slightly.
            let res = (base_resolution / (1u32 << i)).max(2);
            volumes.push(IrradianceVolume::new(bounds, res, res, res, order));
        }
        Self { volumes }
    }

    /// Sample the cascaded volumes at a world position.
    /// Uses the highest-resolution volume that contains the point.
    /// Blends with the next cascade at the boundary.
    pub fn sample(&self, world_pos: Vec3, normal: Vec3) -> Color {
        for (i, vol) in self.volumes.iter().enumerate() {
            if vol.contains(world_pos) {
                let primary = vol.sample(world_pos, normal);

                // Compute blend factor based on distance to volume boundary.
                let ext = vol.bounds.extent();
                let center = vol.bounds.center();
                let dx = ((world_pos.x - center.x).abs() / (ext.x * 0.5)).clamp(0.0, 1.0);
                let dy = ((world_pos.y - center.y).abs() / (ext.y * 0.5)).clamp(0.0, 1.0);
                let dz = ((world_pos.z - center.z).abs() / (ext.z * 0.5)).clamp(0.0, 1.0);
                let edge_factor = dx.max(dy).max(dz);

                // Blend start at 80% from center.
                let blend_start = 0.8;
                if edge_factor > blend_start {
                    if let Some(next) = self.volumes.get(i + 1) {
                        let next_sample = next.sample(world_pos, normal);
                        let t = (edge_factor - blend_start) / (1.0 - blend_start);
                        return primary.lerp(next_sample, t);
                    }
                }
                return primary;
            }
        }

        // Fall back to the largest (last) volume.
        if let Some(last) = self.volumes.last() {
            last.sample(world_pos, normal)
        } else {
            Color::BLACK
        }
    }

    /// Total probe count across all cascades.
    pub fn total_probes(&self) -> usize {
        self.volumes.iter().map(|v| v.count()).sum()
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
    fn sh_order_l1_count() {
        assert_eq!(ShOrder::L1.coeff_count(), 4);
    }

    #[test]
    fn sh_order_l2_count() {
        assert_eq!(ShOrder::L2.coeff_count(), 9);
    }

    #[test]
    fn sh_coeffs_zero() {
        let c = ShCoeffs::zero(ShOrder::L2);
        assert_eq!(c.coeffs.len(), 9);
        assert!(approx(c.evaluate(Vec3::new(0.0, 0.0, 1.0)), 0.0));
    }

    #[test]
    fn sh_coeffs_band0_constant() {
        let mut c = ShCoeffs::zero(ShOrder::L1);
        c.coeffs[0] = 1.0;
        let v1 = c.evaluate(Vec3::new(1.0, 0.0, 0.0));
        let v2 = c.evaluate(Vec3::new(0.0, 0.0, 1.0));
        assert!(approx(v1, v2));
    }

    #[test]
    fn sh_scale() {
        let mut c = ShCoeffs::zero(ShOrder::L1);
        c.coeffs[0] = 1.0;
        c.scale(3.0);
        assert!(approx(c.coeffs[0], 3.0));
    }

    #[test]
    fn sh_add_scaled() {
        let mut a = ShCoeffs::zero(ShOrder::L1);
        a.coeffs[0] = 1.0;
        let mut b = ShCoeffs::zero(ShOrder::L1);
        b.coeffs[0] = 4.0;
        a.add_scaled(&b, 0.5);
        assert!(approx(a.coeffs[0], 3.0));
    }

    #[test]
    fn aabb_contains() {
        let b = AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        assert!(b.contains(Vec3::new(5.0, 5.0, 5.0)));
        assert!(!b.contains(Vec3::new(-1.0, 5.0, 5.0)));
    }

    #[test]
    fn aabb_center() {
        let b = AABB::new(Vec3::ZERO, Vec3::new(10.0, 20.0, 30.0));
        let c = b.center();
        assert!(approx(c.x, 5.0));
        assert!(approx(c.y, 10.0));
        assert!(approx(c.z, 15.0));
    }

    #[test]
    fn probe_new_valid() {
        let p = IrradianceProbe::new(Vec3::ZERO, ShOrder::L2);
        assert!(p.valid);
    }

    #[test]
    fn probe_invalidate() {
        let mut p = IrradianceProbe::new(Vec3::ZERO, ShOrder::L2);
        p.invalidate();
        assert!(!p.valid);
        let c = p.evaluate(Vec3::new(0.0, 0.0, 1.0));
        assert!(color_approx(c, Color::BLACK));
    }

    #[test]
    fn volume_count() {
        let v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            3, 3, 3,
            ShOrder::L2,
        );
        assert_eq!(v.count(), 27);
    }

    #[test]
    fn volume_valid_count() {
        let mut v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L1,
        );
        assert_eq!(v.valid_count(), 8);
        v.get_mut(0, 0, 0).unwrap().invalidate();
        assert_eq!(v.valid_count(), 7);
        assert_eq!(v.invalid_count(), 1);
    }

    #[test]
    fn volume_contains() {
        let v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L1,
        );
        assert!(v.contains(Vec3::new(5.0, 5.0, 5.0)));
        assert!(!v.contains(Vec3::new(-1.0, 5.0, 5.0)));
    }

    #[test]
    fn volume_spacing() {
        let v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            3, 3, 3,
            ShOrder::L1,
        );
        let s = v.spacing();
        assert!(approx(s.x, 5.0));
        assert!(approx(s.y, 5.0));
        assert!(approx(s.z, 5.0));
    }

    #[test]
    fn volume_sample_zero_probes() {
        let v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L2,
        );
        let c = v.sample(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(color_approx(c, Color::BLACK));
    }

    #[test]
    fn volume_sample_with_data() {
        let mut v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L1,
        );
        // Set constant illumination on all probes.
        let band0 = 1.0 / 0.282094791773878;
        for p in &mut v.probes {
            p.sh_r.coeffs[0] = band0;
            p.sh_g.coeffs[0] = band0 * 0.5;
        }
        let c = v.sample(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(c.r > 0.0);
        assert!(c.g > 0.0);
    }

    #[test]
    fn volume_sample_skips_invalid() {
        let mut v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L1,
        );
        let band0 = 1.0 / 0.282094791773878;
        for p in &mut v.probes {
            p.sh_r.coeffs[0] = band0;
        }
        // Invalidate half the probes.
        v.get_mut(0, 0, 0).unwrap().invalidate();
        v.get_mut(1, 0, 0).unwrap().invalidate();
        v.get_mut(0, 1, 0).unwrap().invalidate();
        v.get_mut(1, 1, 0).unwrap().invalidate();
        let c = v.sample(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, 1.0));
        // Should still get light from valid probes.
        assert!(c.r > 0.0);
    }

    #[test]
    fn volume_sample_outside_clamps() {
        let v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L1,
        );
        // Should not panic.
        let _c = v.sample(Vec3::new(-100.0, -100.0, -100.0), Vec3::new(0.0, 1.0, 0.0));
    }

    #[test]
    fn cascaded_create() {
        let cv = CascadedIrradianceVolumes::create_cascaded(
            Vec3::ZERO,
            3,
            10.0,
            8,
            ShOrder::L1,
        );
        assert_eq!(cv.volumes.len(), 3);
        // First volume should be smallest.
        let ext0 = cv.volumes[0].bounds.extent();
        let ext2 = cv.volumes[2].bounds.extent();
        assert!(ext0.x < ext2.x);
    }

    #[test]
    fn cascaded_total_probes() {
        let cv = CascadedIrradianceVolumes::create_cascaded(
            Vec3::ZERO,
            2,
            10.0,
            4,
            ShOrder::L1,
        );
        assert!(cv.total_probes() > 0);
    }

    #[test]
    fn cascaded_sample_center() {
        let cv = CascadedIrradianceVolumes::create_cascaded(
            Vec3::ZERO,
            2,
            10.0,
            4,
            ShOrder::L1,
        );
        // Zero probes, should return black.
        let c = cv.sample(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert!(color_approx(c, Color::BLACK));
    }

    #[test]
    fn cascaded_sample_outside_all() {
        let cv = CascadedIrradianceVolumes::create_cascaded(
            Vec3::ZERO,
            2,
            10.0,
            4,
            ShOrder::L1,
        );
        let _c = cv.sample(Vec3::new(1000.0, 1000.0, 1000.0), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn cascaded_empty() {
        let cv = CascadedIrradianceVolumes::new();
        let c = cv.sample(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert!(color_approx(c, Color::BLACK));
    }

    #[test]
    fn color_lerp() {
        let c = Color::BLACK.lerp(Color::WHITE, 0.5);
        assert!(approx(c.r, 0.5));
    }

    #[test]
    fn volume_get_out_of_bounds() {
        let v = IrradianceVolume::new(
            AABB::new(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0)),
            2, 2, 2,
            ShOrder::L1,
        );
        assert!(v.get(5, 0, 0).is_none());
    }

    #[test]
    fn cascaded_higher_res_first() {
        let cv = CascadedIrradianceVolumes::create_cascaded(
            Vec3::ZERO,
            3,
            10.0,
            16,
            ShOrder::L2,
        );
        // First volume should have higher resolution.
        assert!(cv.volumes[0].res_x >= cv.volumes[1].res_x);
        assert!(cv.volumes[1].res_x >= cv.volumes[2].res_x);
    }
}
