// volumetric_fog.rs — Volumetric fog / participating media with froxel grid
// Part of joule-web: Particles & VFX cluster

/// 3D vector.
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

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-10 { Self::ZERO } else { Self { x: self.x / l, y: self.y / l, z: self.z / l } }
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y, z: self.z + r.z } }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y, z: self.z - r.z } }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
}

/// RGB color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color3 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Color3 {
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0 };
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0 };

    pub fn new(r: f32, g: f32, b: f32) -> Self { Self { r, g, b } }

    pub fn scale(self, s: f32) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s }
    }

    pub fn add(self, o: Self) -> Self {
        Self { r: self.r + o.r, g: self.g + o.g, b: self.b + o.b }
    }

    pub fn mul(self, o: Self) -> Self {
        Self { r: self.r * o.r, g: self.g * o.g, b: self.b * o.b }
    }

    pub fn lerp(self, o: Self, t: f32) -> Self {
        Self {
            r: self.r + (o.r - self.r) * t,
            g: self.g + (o.g - self.g) * t,
            b: self.b + (o.b - self.b) * t,
        }
    }
}

/// RGBA color for final output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color4 {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color4 {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
    pub fn transparent() -> Self { Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 } }
}

/// Data stored per froxel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FroxelData {
    pub density: f32,
    pub scattering_color: Color3,
    pub absorption: f32,
    pub in_scatter: Color3,
}

impl FroxelData {
    pub fn empty() -> Self {
        Self {
            density: 0.0,
            scattering_color: Color3::BLACK,
            absorption: 0.0,
            in_scatter: Color3::BLACK,
        }
    }

    /// Total extinction = scattering + absorption.
    pub fn extinction(&self) -> f32 {
        self.density + self.absorption
    }
}

/// Phase function type for light scattering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseFunction {
    /// Uniform scattering in all directions.
    Isotropic,
    /// Henyey-Greenstein with asymmetry parameter g in (-1, 1).
    HenyeyGreenstein { g_param: f32 },
}

impl PhaseFunction {
    /// Evaluate the phase function given cos(theta) between view and light directions.
    pub fn evaluate(&self, cos_theta: f32) -> f32 {
        match self {
            PhaseFunction::Isotropic => {
                1.0 / (4.0 * std::f32::consts::PI)
            }
            PhaseFunction::HenyeyGreenstein { g_param } => {
                let g2 = g_param * g_param;
                let denom = 1.0 + g2 - 2.0 * g_param * cos_theta;
                if denom < 1e-9 { return 1.0 / (4.0 * std::f32::consts::PI); }
                (1.0 - g2) / (4.0 * std::f32::consts::PI * denom * denom.sqrt())
            }
        }
    }
}

/// Fog source.
#[derive(Debug, Clone, PartialEq)]
pub enum FogSource {
    /// Uniform fog with exponential height falloff.
    HeightExponential {
        base_density: f32,
        height_falloff: f32,
        base_height: f32,
        scattering_color: Color3,
    },
    /// Local box-shaped fog volume.
    BoxVolume {
        center: Vec3,
        half_extents: Vec3,
        density: f32,
        scattering_color: Color3,
    },
    /// Local sphere-shaped fog volume.
    SphereVolume {
        center: Vec3,
        radius: f32,
        density: f32,
        scattering_color: Color3,
    },
}

/// A point light that scatters through fog.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FogLight {
    pub position: Vec3,
    pub color: Color3,
    pub intensity: f32,
    pub range: f32,
}

/// Configuration for the volumetric fog system.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumetricFogConfig {
    /// Grid resolution (X, Y, Z).
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    /// Near and far plane for the froxel grid.
    pub near_plane: f32,
    pub far_plane: f32,
    /// Temporal blend factor for reprojection (0=no blend, 1=full history).
    pub temporal_blend: f32,
    pub phase_function: PhaseFunction,
    /// Jitter offset for sample positions (0 = no jitter).
    pub jitter_scale: f32,
}

impl Default for VolumetricFogConfig {
    fn default() -> Self {
        Self {
            grid_x: 32,
            grid_y: 32,
            grid_z: 64,
            near_plane: 0.1,
            far_plane: 100.0,
            temporal_blend: 0.9,
            phase_function: PhaseFunction::HenyeyGreenstein { g_param: 0.5 },
            jitter_scale: 0.5,
        }
    }
}

/// The froxel grid.
pub struct FroxelGrid {
    config: VolumetricFogConfig,
    data: Vec<FroxelData>,
    history: Vec<FroxelData>,
}

impl FroxelGrid {
    pub fn new(config: VolumetricFogConfig) -> Self {
        let count = (config.grid_x * config.grid_y * config.grid_z) as usize;
        Self {
            data: vec![FroxelData::empty(); count],
            history: vec![FroxelData::empty(); count],
            config,
        }
    }

    pub fn config(&self) -> &VolumetricFogConfig {
        &self.config
    }

    fn index(&self, x: u32, y: u32, z: u32) -> usize {
        ((z * self.config.grid_y + y) * self.config.grid_x + x) as usize
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> FroxelData {
        if x >= self.config.grid_x || y >= self.config.grid_y || z >= self.config.grid_z {
            return FroxelData::empty();
        }
        self.data[self.index(x, y, z)]
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, val: FroxelData) {
        if x < self.config.grid_x && y < self.config.grid_y && z < self.config.grid_z {
            let idx = self.index(x, y, z);
            self.data[idx] = val;
        }
    }

    pub fn clear(&mut self) {
        for d in &mut self.data {
            *d = FroxelData::empty();
        }
    }

    pub fn froxel_count(&self) -> usize {
        (self.config.grid_x * self.config.grid_y * self.config.grid_z) as usize
    }

    /// Map a froxel coordinate to a world-space position (center of froxel).
    /// Uses exponential depth slicing for better near-plane detail.
    pub fn froxel_to_world(&self, x: u32, y: u32, z: u32) -> Vec3 {
        let fx = (x as f32 + 0.5) / self.config.grid_x as f32;
        let fy = (y as f32 + 0.5) / self.config.grid_y as f32;
        let fz = (z as f32 + 0.5) / self.config.grid_z as f32;
        // Exponential depth distribution
        let depth = self.config.near_plane
            * (self.config.far_plane / self.config.near_plane).powf(fz);
        // Map XY to [-1, 1] frustum space * depth
        Vec3::new(
            (fx * 2.0 - 1.0) * depth,
            (fy * 2.0 - 1.0) * depth,
            depth,
        )
    }

    /// Inject fog sources into the froxel grid.
    pub fn inject_sources(&mut self, sources: &[FogSource]) {
        self.clear();
        for z in 0..self.config.grid_z {
            for y in 0..self.config.grid_y {
                for x in 0..self.config.grid_x {
                    let world_pos = self.froxel_to_world(x, y, z);
                    let mut froxel = FroxelData::empty();

                    for source in sources {
                        match source {
                            FogSource::HeightExponential {
                                base_density, height_falloff, base_height, scattering_color,
                            } => {
                                let h = world_pos.y - *base_height;
                                let d = base_density * (-h * height_falloff).exp();
                                froxel.density += d;
                                froxel.scattering_color = froxel.scattering_color.add(
                                    scattering_color.scale(d),
                                );
                            }
                            FogSource::BoxVolume { center, half_extents, density, scattering_color } => {
                                let rel = world_pos - *center;
                                if rel.x.abs() <= half_extents.x
                                    && rel.y.abs() <= half_extents.y
                                    && rel.z.abs() <= half_extents.z
                                {
                                    froxel.density += density;
                                    froxel.scattering_color = froxel.scattering_color.add(
                                        scattering_color.scale(*density),
                                    );
                                }
                            }
                            FogSource::SphereVolume { center, radius, density, scattering_color } => {
                                let dist = (world_pos - *center).length();
                                if dist <= *radius {
                                    let falloff = 1.0 - dist / *radius;
                                    let d = density * falloff;
                                    froxel.density += d;
                                    froxel.scattering_color = froxel.scattering_color.add(
                                        scattering_color.scale(d),
                                    );
                                }
                            }
                        }
                    }

                    let idx = self.index(x, y, z);
                    self.data[idx] = froxel;
                }
            }
        }
    }

    /// Apply in-scattering from lights.
    pub fn apply_lighting(&mut self, lights: &[FogLight], view_dir: Vec3) {
        let phase = &self.config.phase_function;
        for z in 0..self.config.grid_z {
            for y in 0..self.config.grid_y {
                for x in 0..self.config.grid_x {
                    let idx = self.index(x, y, z);
                    if self.data[idx].density < 1e-6 {
                        continue;
                    }
                    let world_pos = self.froxel_to_world(x, y, z);
                    let mut in_scatter = Color3::BLACK;

                    for light in lights {
                        let to_light = light.position - world_pos;
                        let dist = to_light.length();
                        if dist > light.range || dist < 1e-6 {
                            continue;
                        }
                        let light_dir = to_light.normalized();
                        let cos_theta = view_dir.dot(light_dir);
                        let phase_val = phase.evaluate(cos_theta);

                        let attenuation = 1.0 / (dist * dist + 1.0);
                        let contribution = light.color.scale(
                            light.intensity * attenuation * phase_val * self.data[idx].density,
                        );
                        in_scatter = in_scatter.add(contribution);
                    }

                    self.data[idx].in_scatter = in_scatter;
                }
            }
        }
    }

    /// Temporal reprojection: blend current data with history.
    pub fn temporal_blend(&mut self) {
        let blend = self.config.temporal_blend;
        let count = self.data.len();
        for i in 0..count {
            let current = &self.data[i];
            let prev = &self.history[i];
            let blended = FroxelData {
                density: prev.density * blend + current.density * (1.0 - blend),
                scattering_color: prev.scattering_color.lerp(current.scattering_color, 1.0 - blend),
                absorption: prev.absorption * blend + current.absorption * (1.0 - blend),
                in_scatter: prev.in_scatter.lerp(current.in_scatter, 1.0 - blend),
            };
            self.data[i] = blended;
        }
        self.history = self.data.clone();
    }

    /// Front-to-back ray march accumulation along a ray.
    /// Returns accumulated (scatter color, transmittance).
    pub fn ray_march(&self, steps: u32, jitter: f32) -> Color4 {
        let mut accum_color = Color3::BLACK;
        let mut transmittance = 1.0f32;
        let step_size = 1.0 / steps as f32;

        for step_i in 0..steps {
            let t = (step_i as f32 + 0.5 + jitter * self.config.jitter_scale) * step_size;
            let z_idx = (t * self.config.grid_z as f32).min(self.config.grid_z as f32 - 1.0) as u32;
            // Sample center column for simplicity
            let cx = self.config.grid_x / 2;
            let cy = self.config.grid_y / 2;
            let froxel = self.get(cx, cy, z_idx);

            let extinction = froxel.extinction() * step_size;
            let scatter = froxel.in_scatter.add(froxel.scattering_color.scale(froxel.density));

            accum_color = accum_color.add(scatter.scale(transmittance * step_size));
            transmittance *= (-extinction).exp();

            if transmittance < 0.001 {
                break;
            }
        }

        Color4::new(accum_color.r, accum_color.g, accum_color.b, 1.0 - transmittance)
    }

    /// Get a depth-slice density for debugging.
    pub fn slice_density(&self, z: u32) -> Vec<f32> {
        let mut result = Vec::with_capacity((self.config.grid_x * self.config.grid_y) as usize);
        for y in 0..self.config.grid_y {
            for x in 0..self.config.grid_x {
                result.push(self.get(x, y, z).density);
            }
        }
        result
    }
}

/// Add jitter from a simple hash.
pub fn jitter_offset(frame: u32) -> f32 {
    let h = frame.wrapping_mul(2654435761);
    (h as f32 / u32::MAX as f32) - 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> VolumetricFogConfig {
        VolumetricFogConfig {
            grid_x: 4,
            grid_y: 4,
            grid_z: 8,
            near_plane: 0.1,
            far_plane: 50.0,
            temporal_blend: 0.5,
            phase_function: PhaseFunction::Isotropic,
            jitter_scale: 0.0,
        }
    }

    #[test]
    fn test_froxel_grid_new() {
        let grid = FroxelGrid::new(small_config());
        assert_eq!(grid.froxel_count(), 4 * 4 * 8);
    }

    #[test]
    fn test_froxel_get_set() {
        let mut grid = FroxelGrid::new(small_config());
        let data = FroxelData {
            density: 0.5,
            scattering_color: Color3::WHITE,
            absorption: 0.1,
            in_scatter: Color3::BLACK,
        };
        grid.set(1, 2, 3, data);
        let got = grid.get(1, 2, 3);
        assert!((got.density - 0.5).abs() < 1e-6);
        assert!((got.absorption - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_froxel_out_of_bounds() {
        let grid = FroxelGrid::new(small_config());
        let got = grid.get(100, 100, 100);
        assert!((got.density).abs() < 1e-6);
    }

    #[test]
    fn test_froxel_clear() {
        let mut grid = FroxelGrid::new(small_config());
        grid.set(0, 0, 0, FroxelData {
            density: 1.0,
            scattering_color: Color3::WHITE,
            absorption: 0.0,
            in_scatter: Color3::BLACK,
        });
        grid.clear();
        assert!((grid.get(0, 0, 0).density).abs() < 1e-6);
    }

    #[test]
    fn test_froxel_to_world_depth() {
        let grid = FroxelGrid::new(small_config());
        let near = grid.froxel_to_world(0, 0, 0);
        let far = grid.froxel_to_world(0, 0, 7);
        // Far froxel should be at greater depth
        assert!(far.z > near.z);
    }

    #[test]
    fn test_froxel_to_world_exponential() {
        let grid = FroxelGrid::new(small_config());
        let z0 = grid.froxel_to_world(2, 2, 0).z;
        let z1 = grid.froxel_to_world(2, 2, 1).z;
        let z2 = grid.froxel_to_world(2, 2, 2).z;
        // Exponential: spacing should increase
        let d01 = z1 - z0;
        let d12 = z2 - z1;
        assert!(d12 > d01);
    }

    #[test]
    fn test_inject_height_fog() {
        let mut grid = FroxelGrid::new(small_config());
        let sources = vec![FogSource::HeightExponential {
            base_density: 1.0,
            height_falloff: 0.5,
            base_height: 0.0,
            scattering_color: Color3::WHITE,
        }];
        grid.inject_sources(&sources);
        // Some froxels should have density
        let mut has_density = false;
        for z in 0..8 {
            if grid.get(2, 2, z).density > 1e-6 {
                has_density = true;
                break;
            }
        }
        assert!(has_density);
    }

    #[test]
    fn test_inject_box_volume() {
        let cfg = small_config();
        let mut grid = FroxelGrid::new(cfg);
        let center = grid.froxel_to_world(2, 2, 4);
        let sources = vec![FogSource::BoxVolume {
            center,
            half_extents: Vec3::new(100.0, 100.0, 100.0),
            density: 0.5,
            scattering_color: Color3::new(0.5, 0.5, 0.5),
        }];
        grid.inject_sources(&sources);
        let d = grid.get(2, 2, 4).density;
        assert!(d > 0.0);
    }

    #[test]
    fn test_inject_sphere_volume() {
        let cfg = small_config();
        let mut grid = FroxelGrid::new(cfg);
        let center = grid.froxel_to_world(2, 2, 4);
        let sources = vec![FogSource::SphereVolume {
            center,
            radius: 1000.0,
            density: 1.0,
            scattering_color: Color3::WHITE,
        }];
        grid.inject_sources(&sources);
        assert!(grid.get(2, 2, 4).density > 0.0);
    }

    #[test]
    fn test_phase_isotropic() {
        let p = PhaseFunction::Isotropic;
        let v = p.evaluate(0.5);
        assert!((v - 1.0 / (4.0 * std::f32::consts::PI)).abs() < 1e-5);
    }

    #[test]
    fn test_phase_hg_forward() {
        let p = PhaseFunction::HenyeyGreenstein { g_param: 0.8 };
        let forward = p.evaluate(1.0);
        let backward = p.evaluate(-1.0);
        // Forward scattering (g>0) should have more intensity forward
        assert!(forward > backward);
    }

    #[test]
    fn test_phase_hg_zero_is_isotropic() {
        let hg = PhaseFunction::HenyeyGreenstein { g_param: 0.0 };
        let iso = PhaseFunction::Isotropic;
        let v1 = hg.evaluate(0.5);
        let v2 = iso.evaluate(0.5);
        assert!((v1 - v2).abs() < 1e-4);
    }

    #[test]
    fn test_apply_lighting() {
        let mut grid = FroxelGrid::new(small_config());
        // Set some density
        let idx = grid.index(2, 2, 4);
        grid.data[idx].density = 0.5;
        grid.data[idx].scattering_color = Color3::WHITE;

        let lights = vec![FogLight {
            position: grid.froxel_to_world(2, 2, 4),
            color: Color3::WHITE,
            intensity: 10.0,
            range: 1000.0,
        }];

        grid.apply_lighting(&lights, Vec3::new(0.0, 0.0, 1.0));
        // In-scatter might be near zero since the light is at the froxel itself
        // (distance ~ 0), but the formula handles dist < 1e-6 by skipping
        // Let's test a light further away
        let lights2 = vec![FogLight {
            position: grid.froxel_to_world(2, 2, 4) + Vec3::new(5.0, 0.0, 0.0),
            color: Color3::WHITE,
            intensity: 100.0,
            range: 1000.0,
        }];
        grid.apply_lighting(&lights2, Vec3::new(0.0, 0.0, 1.0));
        let inscatter = grid.get(2, 2, 4).in_scatter;
        assert!(inscatter.r > 0.0 || inscatter.g > 0.0 || inscatter.b > 0.0);
    }

    #[test]
    fn test_temporal_blend() {
        let mut grid = FroxelGrid::new(VolumetricFogConfig {
            temporal_blend: 0.5,
            ..small_config()
        });
        // Set history to density=1.0
        let idx = grid.index(2, 2, 4);
        grid.history[idx].density = 1.0;
        // Set current to density=0.0
        grid.data[idx].density = 0.0;
        grid.temporal_blend();
        // Should be blended to 0.5
        assert!((grid.get(2, 2, 4).density - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_ray_march_empty() {
        let grid = FroxelGrid::new(small_config());
        let result = grid.ray_march(16, 0.0);
        // Empty fog => no color, no opacity
        assert!((result.a).abs() < 1e-3);
    }

    #[test]
    fn test_ray_march_with_fog() {
        let mut grid = FroxelGrid::new(small_config());
        // Fill all center-column froxels with density
        for z in 0..8 {
            let idx = grid.index(2, 2, z);
            grid.data[idx].density = 0.5;
            grid.data[idx].scattering_color = Color3::new(0.8, 0.6, 0.4);
        }
        let result = grid.ray_march(8, 0.0);
        assert!(result.a > 0.0);
        assert!(result.r > 0.0);
    }

    #[test]
    fn test_slice_density() {
        let mut grid = FroxelGrid::new(small_config());
        let idx = grid.index(1, 1, 3);
        grid.data[idx].density = 0.77;
        let slice = grid.slice_density(3);
        assert_eq!(slice.len(), 16); // 4*4
        assert!((slice[1 * 4 + 1] - 0.77).abs() < 1e-6);
    }

    #[test]
    fn test_froxel_extinction() {
        let f = FroxelData {
            density: 0.3,
            absorption: 0.2,
            scattering_color: Color3::BLACK,
            in_scatter: Color3::BLACK,
        };
        assert!((f.extinction() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_jitter_offset_range() {
        for frame in 0..100 {
            let j = jitter_offset(frame);
            assert!(j >= -0.5 && j <= 0.5);
        }
    }

    #[test]
    fn test_color3_operations() {
        let a = Color3::new(0.5, 0.3, 0.1);
        let b = Color3::new(0.2, 0.4, 0.6);
        let sum = a.add(b);
        assert!((sum.r - 0.7).abs() < 1e-6);
        let prod = a.mul(b);
        assert!((prod.r - 0.1).abs() < 1e-6);
        let scaled = a.scale(2.0);
        assert!((scaled.r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_color3_lerp() {
        let a = Color3::BLACK;
        let b = Color3::WHITE;
        let mid = a.lerp(b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_multiple_sources_additive() {
        let mut grid = FroxelGrid::new(small_config());
        let center = grid.froxel_to_world(2, 2, 4);
        let sources = vec![
            FogSource::SphereVolume {
                center,
                radius: 1000.0,
                density: 0.3,
                scattering_color: Color3::WHITE,
            },
            FogSource::SphereVolume {
                center,
                radius: 1000.0,
                density: 0.2,
                scattering_color: Color3::WHITE,
            },
        ];
        grid.inject_sources(&sources);
        let d = grid.get(2, 2, 4).density;
        assert!(d > 0.4);
    }

    #[test]
    fn test_default_config() {
        let cfg = VolumetricFogConfig::default();
        assert_eq!(cfg.grid_x, 32);
        assert_eq!(cfg.grid_z, 64);
    }
}
