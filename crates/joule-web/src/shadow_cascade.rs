//! Cascaded Shadow Maps (CSM) for directional lights.
//!
//! Frustum splitting (uniform, logarithmic, practical/PSSM), per-cascade
//! light-space orthographic matrices with texel-grid snapping to eliminate
//! shimmer, cascade selection per fragment, cascade blending at boundaries,
//! shadow bias (constant + slope-based), and PCF (Percentage Closer
//! Filtering) soft shadows with configurable kernel.

use std::f64::consts::PI;

// ── Vector / Matrix types ──────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const UP: Self = Self { x: 0.0, y: 1.0, z: 0.0 };

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
    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-12 { return Self::ZERO; }
        Self { x: self.x / l, y: self.y / l, z: self.z / l }
    }
    pub fn sub(self, o: Self) -> Self { Self { x: self.x - o.x, y: self.y - o.y, z: self.z - o.z } }
    pub fn add(self, o: Self) -> Self { Self { x: self.x + o.x, y: self.y + o.y, z: self.z + o.z } }
    pub fn scale(self, s: f64) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self { Self { x, y, z, w } }
}

/// Row-major 4×4 matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub m: [f64; 16],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    };

    pub fn look_at(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = center.sub(eye).normalized();
        let s = f.cross(up).normalized();
        let u = s.cross(f);
        Self {
            m: [
                s.x,  s.y,  s.z, -s.dot(eye),
                u.x,  u.y,  u.z, -u.dot(eye),
               -f.x, -f.y, -f.z,  f.dot(eye),
                0.0,  0.0,  0.0,  1.0,
            ],
        }
    }

    pub fn ortho(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self {
        let w = right - left;
        let h = top - bottom;
        let d = far - near;
        Self {
            m: [
                2.0 / w, 0.0,     0.0,      -(right + left) / w,
                0.0,     2.0 / h, 0.0,      -(top + bottom) / h,
                0.0,     0.0,    -2.0 / d,  -(far + near) / d,
                0.0,     0.0,     0.0,       1.0,
            ],
        }
    }

    pub fn mul(&self, o: &Self) -> Self {
        let mut r = [0.0f64; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 { sum += self.m[row * 4 + k] * o.m[k * 4 + col]; }
                r[row * 4 + col] = sum;
            }
        }
        Self { m: r }
    }

    pub fn transform_vec4(&self, v: Vec4) -> Vec4 {
        Vec4 {
            x: self.m[0] * v.x + self.m[1] * v.y + self.m[2] * v.z + self.m[3] * v.w,
            y: self.m[4] * v.x + self.m[5] * v.y + self.m[6] * v.z + self.m[7] * v.w,
            z: self.m[8] * v.x + self.m[9] * v.y + self.m[10] * v.z + self.m[11] * v.w,
            w: self.m[12] * v.x + self.m[13] * v.y + self.m[14] * v.z + self.m[15] * v.w,
        }
    }
}

// ── Frustum split schemes ──────────────────────────────────────

/// Split scheme for distributing cascade boundaries.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplitScheme {
    /// Even distribution of depth range.
    Uniform,
    /// Logarithmic distribution (better quality near camera).
    Logarithmic,
    /// Practical (PSSM): blend between uniform and log with `lambda`.
    Practical { lambda: f64 },
}

/// Compute split distances for a given scheme.
pub fn compute_splits(scheme: SplitScheme, cascade_count: u32, near: f64, far: f64) -> Vec<f64> {
    let n = near.max(1e-6);
    let mut splits = Vec::with_capacity(cascade_count as usize + 1);

    for i in 0..=cascade_count {
        let t = i as f64 / cascade_count as f64;
        let val = match scheme {
            SplitScheme::Uniform => {
                n + (far - n) * t
            }
            SplitScheme::Logarithmic => {
                n * (far / n).powf(t)
            }
            SplitScheme::Practical { lambda } => {
                let lambda = lambda.clamp(0.0, 1.0);
                let uni = n + (far - n) * t;
                let log = n * (far / n).powf(t);
                lambda * log + (1.0 - lambda) * uni
            }
        };
        splits.push(val);
    }
    splits
}

// ── Shadow bias ────────────────────────────────────────────────

/// Shadow bias configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShadowBias {
    /// Constant depth offset.
    pub constant: f64,
    /// Slope-scaled bias (multiplied by the surface slope).
    pub slope_scale: f64,
}

impl ShadowBias {
    pub fn new(constant: f64, slope_scale: f64) -> Self { Self { constant, slope_scale } }

    /// Compute total bias given the surface normal and light direction.
    pub fn total(&self, surface_normal: Vec3, light_dir: Vec3) -> f64 {
        let cos_angle = surface_normal.normalized().dot(light_dir.normalized()).abs();
        let slope = if cos_angle > 1e-6 { (1.0 - cos_angle * cos_angle).sqrt() / cos_angle } else { 1e6 };
        self.constant + self.slope_scale * slope
    }
}

// ── PCF kernel ─────────────────────────────────────────────────

/// PCF (Percentage Closer Filtering) configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct PcfKernel {
    /// Kernel size (e.g., 3 = 3×3 samples).
    pub size: u32,
    /// Sample offsets (in texel space) and weights.
    pub samples: Vec<(f64, f64, f64)>,
}

impl PcfKernel {
    /// Create a uniform box kernel.
    pub fn box_kernel(size: u32) -> Self {
        let s = size.max(1);
        let half = s as f64 / 2.0;
        let count = (s * s) as f64;
        let weight = 1.0 / count;
        let mut samples = Vec::with_capacity((s * s) as usize);
        for y in 0..s {
            for x in 0..s {
                let ox = x as f64 - half + 0.5;
                let oy = y as f64 - half + 0.5;
                samples.push((ox, oy, weight));
            }
        }
        Self { size: s, samples }
    }

    /// Create a Gaussian-weighted kernel.
    pub fn gaussian(size: u32, sigma: f64) -> Self {
        let s = size.max(1);
        let half = s as f64 / 2.0;
        let mut samples = Vec::with_capacity((s * s) as usize);
        let mut total_weight = 0.0;
        for y in 0..s {
            for x in 0..s {
                let ox = x as f64 - half + 0.5;
                let oy = y as f64 - half + 0.5;
                let w = (-(ox * ox + oy * oy) / (2.0 * sigma * sigma)).exp();
                samples.push((ox, oy, w));
                total_weight += w;
            }
        }
        // Normalize weights.
        if total_weight > 1e-12 {
            for s in &mut samples { s.2 /= total_weight; }
        }
        Self { size: s, samples }
    }

    /// Evaluate PCF: given a depth comparison function (returns 1.0 if lit,
    /// 0.0 if shadowed) at each sample offset, return the averaged shadow factor.
    pub fn evaluate<F>(&self, compare: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let mut result = 0.0;
        for (ox, oy, w) in &self.samples {
            result += compare(*ox, *oy) * w;
        }
        result
    }
}

// ── Cascade data ───────────────────────────────────────────────

/// Data for a single cascade slice.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadeSlice {
    /// Index of this cascade.
    pub index: u32,
    /// Near depth boundary.
    pub near: f64,
    /// Far depth boundary.
    pub far: f64,
    /// Light-space projection × view matrix.
    pub light_matrix: Mat4,
    /// Shadow map resolution for this cascade.
    pub resolution: u32,
}

// ── CascadedShadowMap ──────────────────────────────────────────

/// Cascaded shadow map system for a directional light.
#[derive(Debug, Clone, PartialEq)]
pub struct CascadedShadowMap {
    /// Light direction (normalized, toward surfaces).
    pub light_direction: Vec3,
    /// Split scheme.
    pub split_scheme: SplitScheme,
    /// Split distances (len = cascade_count + 1).
    pub split_distances: Vec<f64>,
    /// Per-cascade data.
    pub cascades: Vec<CascadeSlice>,
    /// Shadow bias configuration.
    pub bias: ShadowBias,
    /// Blend distance at cascade boundaries.
    pub blend_distance: f64,
    /// Shadow map resolution (base).
    pub resolution: u32,
}

impl CascadedShadowMap {
    /// Build a CSM system.
    pub fn new(
        light_direction: Vec3,
        cascade_count: u32,
        near: f64,
        far: f64,
        scheme: SplitScheme,
        resolution: u32,
    ) -> Self {
        let splits = compute_splits(scheme, cascade_count, near, far);
        Self {
            light_direction: light_direction.normalized(),
            split_scheme: scheme,
            split_distances: splits,
            cascades: Vec::new(),
            bias: ShadowBias::new(0.005, 0.05),
            blend_distance: 2.0,
            resolution,
        }
    }

    pub fn with_bias(mut self, bias: ShadowBias) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_blend_distance(mut self, dist: f64) -> Self {
        self.blend_distance = dist.max(0.0);
        self
    }

    /// Compute cascade light-space matrices for a scene bounding sphere.
    /// Populates `self.cascades`.
    pub fn compute_matrices(&mut self, scene_center: Vec3, scene_radius: f64) {
        let count = self.split_distances.len().saturating_sub(1);
        self.cascades.clear();
        self.cascades.reserve(count);

        let up = if self.light_direction.cross(Vec3::UP).length() < 1e-6 {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::UP
        };

        for i in 0..count {
            let near_split = self.split_distances[i];
            let far_split = self.split_distances[i + 1];
            let mid = (near_split + far_split) * 0.5;
            let cascade_radius = ((far_split - near_split) * 0.5).max(scene_radius * 0.05);

            // View along light direction.
            let cascade_center = scene_center.add(self.light_direction.scale(-mid));
            let light_pos = cascade_center.sub(self.light_direction.scale(cascade_radius * 2.0));
            let view = Mat4::look_at(light_pos, cascade_center, up);
            let proj = Mat4::ortho(
                -cascade_radius, cascade_radius,
                -cascade_radius, cascade_radius,
                0.0, cascade_radius * 4.0,
            );

            let light_matrix = self.snap_to_texel(proj.mul(&view), cascade_radius);

            self.cascades.push(CascadeSlice {
                index: i as u32,
                near: near_split,
                far: far_split,
                light_matrix,
                resolution: self.resolution,
            });
        }
    }

    /// Snap the light-space matrix to texel boundaries to prevent shadow shimmer.
    fn snap_to_texel(&self, mat: Mat4, radius: f64) -> Mat4 {
        if self.resolution == 0 { return mat; }
        let texel_size = 2.0 * radius / self.resolution as f64;
        if texel_size < 1e-12 { return mat; }

        let mut result = mat;
        // Snap the translation components (indices 3, 7) to texel grid.
        result.m[3] = (result.m[3] / texel_size).floor() * texel_size;
        result.m[7] = (result.m[7] / texel_size).floor() * texel_size;
        result
    }

    /// Select which cascade a fragment belongs to based on its view-space depth.
    /// Returns the cascade index, or None if outside all cascades.
    pub fn select_cascade(&self, depth: f64) -> Option<u32> {
        for c in &self.cascades {
            if depth >= c.near && depth < c.far {
                return Some(c.index);
            }
        }
        // Check last cascade inclusively.
        if let Some(last) = self.cascades.last() {
            if (depth - last.far).abs() < 1e-6 {
                return Some(last.index);
            }
        }
        None
    }

    /// Compute the blend factor for cascade boundary blending.
    /// Returns a value in [0, 1] where 1 means fully in the current cascade
    /// and < 1 means blending with the next cascade.
    pub fn cascade_blend_factor(&self, depth: f64, cascade_index: u32) -> f64 {
        let idx = cascade_index as usize;
        if idx >= self.cascades.len() { return 1.0; }
        let cascade = &self.cascades[idx];
        let dist_to_far = cascade.far - depth;
        if dist_to_far <= 0.0 { return 0.0; }
        if self.blend_distance <= 0.0 || dist_to_far >= self.blend_distance {
            return 1.0;
        }
        dist_to_far / self.blend_distance
    }

    /// Compute shadow bias for a fragment.
    pub fn compute_bias(&self, surface_normal: Vec3) -> f64 {
        self.bias.total(surface_normal, self.light_direction)
    }

    /// Number of cascades.
    pub fn cascade_count(&self) -> u32 {
        self.cascades.len() as u32
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn approx(a: f64, b: f64) -> bool { (a - b).abs() < EPS }

    #[test]
    fn splits_uniform() {
        let s = compute_splits(SplitScheme::Uniform, 4, 1.0, 100.0);
        assert_eq!(s.len(), 5);
        assert!(approx(s[0], 1.0));
        assert!(approx(s[4], 100.0));
        let step = s[1] - s[0];
        for i in 1..4 {
            assert!((s[i + 1] - s[i] - step).abs() < 0.01);
        }
    }

    #[test]
    fn splits_logarithmic() {
        let s = compute_splits(SplitScheme::Logarithmic, 3, 1.0, 1000.0);
        assert_eq!(s.len(), 4);
        assert!(approx(s[0], 1.0));
        assert!(approx(s[3], 1000.0));
        assert!(approx(s[1], 10.0));
        assert!(approx(s[2], 100.0));
    }

    #[test]
    fn splits_practical_lambda0() {
        let uni = compute_splits(SplitScheme::Uniform, 3, 1.0, 100.0);
        let prac = compute_splits(SplitScheme::Practical { lambda: 0.0 }, 3, 1.0, 100.0);
        for (a, b) in uni.iter().zip(prac.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn splits_practical_lambda1() {
        let log = compute_splits(SplitScheme::Logarithmic, 3, 1.0, 100.0);
        let prac = compute_splits(SplitScheme::Practical { lambda: 1.0 }, 3, 1.0, 100.0);
        for (a, b) in log.iter().zip(prac.iter()) {
            assert!(approx(*a, *b));
        }
    }

    #[test]
    fn splits_monotonically_increasing() {
        let s = compute_splits(SplitScheme::Practical { lambda: 0.5 }, 5, 0.1, 500.0);
        for i in 0..s.len() - 1 {
            assert!(s[i] < s[i + 1], "splits must be monotonically increasing");
        }
    }

    #[test]
    fn shadow_bias_constant_only() {
        let b = ShadowBias::new(0.01, 0.0);
        let t = b.total(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        assert!(approx(t, 0.01));
    }

    #[test]
    fn shadow_bias_slope() {
        let b = ShadowBias::new(0.0, 1.0);
        // Grazing angle: surface normal nearly perpendicular to light.
        let t = b.total(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        // cos_angle ≈ 0, slope → large.
        assert!(t > 1.0);
    }

    #[test]
    fn shadow_bias_head_on() {
        let b = ShadowBias::new(0.0, 1.0);
        let t = b.total(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert!(approx(t, 0.0));
    }

    #[test]
    fn pcf_box_kernel_sums_to_one() {
        let k = PcfKernel::box_kernel(5);
        let sum: f64 = k.samples.iter().map(|s| s.2).sum();
        assert!(approx(sum, 1.0));
    }

    #[test]
    fn pcf_gaussian_sums_to_one() {
        let k = PcfKernel::gaussian(5, 1.0);
        let sum: f64 = k.samples.iter().map(|s| s.2).sum();
        assert!(approx(sum, 1.0));
    }

    #[test]
    fn pcf_evaluate_all_lit() {
        let k = PcfKernel::box_kernel(3);
        let result = k.evaluate(|_, _| 1.0);
        assert!(approx(result, 1.0));
    }

    #[test]
    fn pcf_evaluate_all_shadowed() {
        let k = PcfKernel::box_kernel(3);
        let result = k.evaluate(|_, _| 0.0);
        assert!(approx(result, 0.0));
    }

    #[test]
    fn pcf_evaluate_half() {
        let k = PcfKernel::box_kernel(3);
        let result = k.evaluate(|x, _| if x < 0.0 { 1.0 } else { 0.0 });
        // 3 of 9 samples have x < 0 (the left column: x offsets are -1.0).
        assert!(result > 0.0 && result < 1.0);
    }

    #[test]
    fn csm_construct() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            4,
            0.1,
            100.0,
            SplitScheme::Practical { lambda: 0.5 },
            1024,
        );
        csm.compute_matrices(Vec3::ZERO, 50.0);
        assert_eq!(csm.cascade_count(), 4);
    }

    #[test]
    fn csm_select_cascade_first() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            3,
            1.0,
            100.0,
            SplitScheme::Uniform,
            512,
        );
        csm.compute_matrices(Vec3::ZERO, 50.0);
        let idx = csm.select_cascade(5.0);
        assert_eq!(idx, Some(0));
    }

    #[test]
    fn csm_select_cascade_last() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            3,
            1.0,
            100.0,
            SplitScheme::Uniform,
            512,
        );
        csm.compute_matrices(Vec3::ZERO, 50.0);
        let idx = csm.select_cascade(90.0);
        assert_eq!(idx, Some(2));
    }

    #[test]
    fn csm_select_cascade_outside() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            3,
            1.0,
            100.0,
            SplitScheme::Uniform,
            512,
        );
        csm.compute_matrices(Vec3::ZERO, 50.0);
        let idx = csm.select_cascade(200.0);
        assert!(idx.is_none());
    }

    #[test]
    fn csm_blend_factor_far_from_boundary() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            2,
            1.0,
            100.0,
            SplitScheme::Uniform,
            512,
        ).with_blend_distance(5.0);
        csm.compute_matrices(Vec3::ZERO, 50.0);
        let f = csm.cascade_blend_factor(10.0, 0);
        assert!(approx(f, 1.0));
    }

    #[test]
    fn csm_blend_factor_at_boundary() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            2,
            1.0,
            100.0,
            SplitScheme::Uniform,
            512,
        ).with_blend_distance(5.0);
        csm.compute_matrices(Vec3::ZERO, 50.0);
        // First cascade goes from 1.0 to 50.5.
        let near_far = csm.cascades[0].far;
        let f = csm.cascade_blend_factor(near_far, 0);
        assert!(approx(f, 0.0));
    }

    #[test]
    fn csm_matrices_are_finite() {
        let mut csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            3,
            0.1,
            500.0,
            SplitScheme::Logarithmic,
            2048,
        );
        csm.compute_matrices(Vec3::ZERO, 100.0);
        for c in &csm.cascades {
            for val in &c.light_matrix.m {
                assert!(val.is_finite(), "matrix element must be finite");
            }
        }
    }

    #[test]
    fn csm_compute_bias() {
        let csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            2,
            1.0,
            100.0,
            SplitScheme::Uniform,
            512,
        ).with_bias(ShadowBias::new(0.005, 0.05));
        let b = csm.compute_bias(Vec3::new(0.0, 1.0, 0.0));
        assert!(b >= 0.005);
    }

    #[test]
    fn pcf_gaussian_center_has_most_weight() {
        let k = PcfKernel::gaussian(5, 1.0);
        // Center sample should have the highest weight.
        let center_idx = 12; // 5×5 grid, center at (2,2) = index 12
        let center_w = k.samples[center_idx].2;
        for (i, s) in k.samples.iter().enumerate() {
            if i != center_idx {
                assert!(center_w >= s.2 - EPS);
            }
        }
    }

    #[test]
    fn snap_to_texel_changes_translation() {
        let csm = CascadedShadowMap::new(
            Vec3::new(0.0, -1.0, 0.0),
            1,
            1.0,
            100.0,
            SplitScheme::Uniform,
            64,
        );
        let mat = Mat4 {
            m: [
                1.0, 0.0, 0.0, 3.14159,
                0.0, 1.0, 0.0, 2.71828,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        };
        let snapped = csm.snap_to_texel(mat, 10.0);
        // Translation should be snapped to multiples of texel_size.
        let texel_size = 2.0 * 10.0 / 64.0;
        let tx_rem = snapped.m[3] % texel_size;
        assert!(tx_rem.abs() < EPS || (texel_size - tx_rem.abs()).abs() < EPS);
    }
}
