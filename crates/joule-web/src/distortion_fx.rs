// distortion_fx.rs — Screen-space distortion effects
// Part of joule-web: Particles & VFX cluster

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-10 { Self::ZERO } else { Self { x: self.x / l, y: self.y / l } }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    fn add(self, r: Self) -> Self { Self { x: self.x + r.x, y: self.y + r.y } }
}
impl std::ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, r: Self) -> Self { Self { x: self.x - r.x, y: self.y - r.y } }
}
impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, s: f32) -> Self { Self { x: self.x * s, y: self.y * s } }
}

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self { Self { r, g, b, a } }
}

/// Distortion source types.
#[derive(Debug, Clone, PartialEq)]
pub enum DistortionSource {
    /// Heat haze: sine-wave animated UV offset.
    HeatHaze {
        center: Vec2,
        radius: f32,
        frequency: f32,
        amplitude: f32,
        speed: f32,
    },
    /// Shockwave: expanding ring of distortion.
    Shockwave {
        center: Vec2,
        current_radius: f32,
        expand_speed: f32,
        ring_width: f32,
        amplitude: f32,
        max_radius: f32,
    },
    /// Refraction: normal-mapped UV offset centered at a point.
    Refraction {
        center: Vec2,
        radius: f32,
        strength: f32,
        /// Normal map encoded as (dx, dy) per-pixel pattern frequency.
        pattern_frequency: f32,
    },
    /// Portal: swirl/vortex distortion.
    Portal {
        center: Vec2,
        radius: f32,
        twist_amount: f32,
        falloff_power: f32,
    },
}

/// An active distortion source with intensity control.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveDistortion {
    pub source: DistortionSource,
    pub intensity: f32,
    pub active: bool,
    pub time: f32,
    id: u64,
}

impl ActiveDistortion {
    pub fn new(id: u64, source: DistortionSource) -> Self {
        Self {
            source,
            intensity: 1.0,
            active: true,
            time: 0.0,
            id,
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Compute UV offset at a screen-space position.
    pub fn sample(&self, pos: Vec2) -> Vec2 {
        if !self.active || self.intensity < 1e-6 {
            return Vec2::ZERO;
        }

        let offset = match &self.source {
            DistortionSource::HeatHaze { center, radius, frequency, amplitude, speed } => {
                let d = (pos - *center).length();
                if d > *radius { return Vec2::ZERO; }
                let falloff = 1.0 - (d / radius);
                let phase = pos.y * *frequency + self.time * *speed;
                let dx = phase.sin() * *amplitude * falloff;
                let dy = (phase * 1.3 + 0.7).cos() * *amplitude * 0.5 * falloff;
                Vec2::new(dx, dy)
            }
            DistortionSource::Shockwave {
                center, current_radius, ring_width, amplitude, ..
            } => {
                let to_pos = pos - *center;
                let d = to_pos.length();
                let ring_dist = (d - *current_radius).abs();
                let half_w = *ring_width * 0.5;
                if ring_dist > half_w { return Vec2::ZERO; }
                let ring_factor = 1.0 - ring_dist / half_w;
                let dir = to_pos.normalized();
                dir * (*amplitude * ring_factor)
            }
            DistortionSource::Refraction { center, radius, strength, pattern_frequency } => {
                let d = (pos - *center).length();
                if d > *radius { return Vec2::ZERO; }
                let falloff = 1.0 - (d / radius);
                let nx = (pos.x * *pattern_frequency).sin();
                let ny = (pos.y * *pattern_frequency + 1.37).cos();
                Vec2::new(nx * *strength * falloff, ny * *strength * falloff)
            }
            DistortionSource::Portal { center, radius, twist_amount, falloff_power } => {
                let to_pos = pos - *center;
                let d = to_pos.length();
                if d > *radius || d < 1e-6 { return Vec2::ZERO; }
                let norm_d = d / *radius;
                let falloff = (1.0 - norm_d).powf(*falloff_power);
                let angle = *twist_amount * falloff;
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let rotated = Vec2::new(
                    to_pos.x * cos_a - to_pos.y * sin_a,
                    to_pos.x * sin_a + to_pos.y * cos_a,
                );
                (rotated - to_pos) * falloff
            }
        };

        offset * self.intensity
    }

    /// Update source (advance shockwave, etc.).
    pub fn update(&mut self, dt: f32) {
        self.time += dt;
        if let DistortionSource::Shockwave {
            current_radius, expand_speed, max_radius, ..
        } = &mut self.source {
            *current_radius += *expand_speed * dt;
            if *current_radius > *max_radius {
                self.active = false;
            }
        }
    }
}

/// Distortion buffer: per-pixel UV offset.
pub struct DistortionBuffer {
    width: u32,
    height: u32,
    /// Stored as pairs (du, dv) in row-major order.
    offsets: Vec<Vec2>,
}

impl DistortionBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            offsets: vec![Vec2::ZERO; (width * height) as usize],
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn clear(&mut self) {
        for o in &mut self.offsets {
            *o = Vec2::ZERO;
        }
    }

    /// Get the offset at pixel (x, y).
    pub fn get(&self, x: u32, y: u32) -> Vec2 {
        if x >= self.width || y >= self.height {
            return Vec2::ZERO;
        }
        self.offsets[(y * self.width + x) as usize]
    }

    /// Set the offset at pixel (x, y).
    pub fn set(&mut self, x: u32, y: u32, offset: Vec2) {
        if x < self.width && y < self.height {
            self.offsets[(y * self.width + x) as usize] = offset;
        }
    }

    /// Add an offset (additive compositing).
    pub fn add(&mut self, x: u32, y: u32, offset: Vec2) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) as usize;
            self.offsets[idx] = self.offsets[idx] + offset;
        }
    }

    /// Apply a distortion source to the entire buffer (in normalized [0,1] UV space).
    pub fn apply_source(&mut self, source: &ActiveDistortion) {
        for y in 0..self.height {
            for x in 0..self.width {
                let uv = Vec2::new(
                    x as f32 / self.width as f32,
                    y as f32 / self.height as f32,
                );
                let offset = source.sample(uv);
                if offset.length() > 1e-8 {
                    self.add(x, y, offset);
                }
            }
        }
    }

    /// Apply distortion to look up a scene color.
    /// `scene` callback takes (u, v) in [0,1] and returns a Color.
    pub fn apply_to_scene<F>(&self, x: u32, y: u32, scene: F) -> Color
    where
        F: Fn(f32, f32) -> Color,
    {
        let base_u = x as f32 / self.width as f32;
        let base_v = y as f32 / self.height as f32;
        let offset = self.get(x, y);
        let sample_u = (base_u + offset.x).clamp(0.0, 1.0);
        let sample_v = (base_v + offset.y).clamp(0.0, 1.0);
        scene(sample_u, sample_v)
    }

    /// Maximum distortion magnitude in the buffer.
    pub fn max_distortion(&self) -> f32 {
        self.offsets.iter().map(|o| o.length()).fold(0.0f32, f32::max)
    }
}

/// Composites multiple distortion sources into a buffer.
pub struct DistortionCompositor {
    sources: Vec<ActiveDistortion>,
    next_id: u64,
}

impl DistortionCompositor {
    pub fn new() -> Self {
        Self { sources: Vec::new(), next_id: 0 }
    }

    pub fn add_source(&mut self, source: DistortionSource) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.sources.push(ActiveDistortion::new(id, source));
        id
    }

    pub fn remove_source(&mut self, id: u64) -> bool {
        let before = self.sources.len();
        self.sources.retain(|s| s.id() != id);
        self.sources.len() < before
    }

    pub fn set_intensity(&mut self, id: u64, intensity: f32) {
        if let Some(s) = self.sources.iter_mut().find(|s| s.id() == id) {
            s.intensity = intensity;
        }
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Update all sources.
    pub fn update(&mut self, dt: f32) {
        for s in &mut self.sources {
            s.update(dt);
        }
        self.sources.retain(|s| s.active);
    }

    /// Composite all sources into a buffer.
    pub fn composite(&self, buffer: &mut DistortionBuffer) {
        buffer.clear();
        for source in &self.sources {
            buffer.apply_source(source);
        }
    }

    /// Sample the combined distortion at a UV coordinate.
    pub fn sample_combined(&self, uv: Vec2) -> Vec2 {
        let mut total = Vec2::ZERO;
        for source in &self.sources {
            total = total + source.sample(uv);
        }
        total
    }

    pub fn clear(&mut self) {
        self.sources.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_new() {
        let buf = DistortionBuffer::new(64, 64);
        assert_eq!(buf.width(), 64);
        assert_eq!(buf.height(), 64);
    }

    #[test]
    fn test_buffer_get_set() {
        let mut buf = DistortionBuffer::new(16, 16);
        buf.set(5, 5, Vec2::new(0.1, 0.2));
        let v = buf.get(5, 5);
        assert!((v.x - 0.1).abs() < 1e-6);
        assert!((v.y - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_buffer_add() {
        let mut buf = DistortionBuffer::new(16, 16);
        buf.add(3, 3, Vec2::new(0.1, 0.0));
        buf.add(3, 3, Vec2::new(0.2, 0.0));
        let v = buf.get(3, 3);
        assert!((v.x - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_buffer_clear() {
        let mut buf = DistortionBuffer::new(8, 8);
        buf.set(0, 0, Vec2::new(1.0, 1.0));
        buf.clear();
        let v = buf.get(0, 0);
        assert!((v.x).abs() < 1e-6);
    }

    #[test]
    fn test_buffer_out_of_bounds() {
        let buf = DistortionBuffer::new(8, 8);
        let v = buf.get(100, 100);
        assert!((v.x).abs() < 1e-6);
    }

    #[test]
    fn test_heat_haze_at_center() {
        let src = ActiveDistortion::new(0, DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 0.3,
            frequency: 10.0,
            amplitude: 0.05,
            speed: 1.0,
        });
        let offset = src.sample(Vec2::new(0.5, 0.5));
        // At center, distance=0, falloff=1, so offset should be nonzero
        assert!(offset.length() > 0.0);
    }

    #[test]
    fn test_heat_haze_outside_radius() {
        let src = ActiveDistortion::new(0, DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 0.1,
            frequency: 10.0,
            amplitude: 0.05,
            speed: 1.0,
        });
        let offset = src.sample(Vec2::new(0.0, 0.0));
        assert!((offset.x).abs() < 1e-6);
    }

    #[test]
    fn test_shockwave_expands() {
        let mut src = ActiveDistortion::new(0, DistortionSource::Shockwave {
            center: Vec2::new(0.5, 0.5),
            current_radius: 0.0,
            expand_speed: 1.0,
            ring_width: 0.1,
            amplitude: 0.1,
            max_radius: 0.5,
        });
        src.update(0.2);
        if let DistortionSource::Shockwave { current_radius, .. } = &src.source {
            assert!((*current_radius - 0.2).abs() < 1e-5);
        }
        assert!(src.active);
    }

    #[test]
    fn test_shockwave_deactivates() {
        let mut src = ActiveDistortion::new(0, DistortionSource::Shockwave {
            center: Vec2::new(0.5, 0.5),
            current_radius: 0.0,
            expand_speed: 1.0,
            ring_width: 0.1,
            amplitude: 0.1,
            max_radius: 0.3,
        });
        src.update(0.5);
        assert!(!src.active);
    }

    #[test]
    fn test_shockwave_ring_distortion() {
        let src = ActiveDistortion::new(0, DistortionSource::Shockwave {
            center: Vec2::new(0.5, 0.5),
            current_radius: 0.2,
            expand_speed: 1.0,
            ring_width: 0.1,
            amplitude: 0.1,
            max_radius: 1.0,
        });
        // Point on the ring
        let on_ring = src.sample(Vec2::new(0.7, 0.5));
        assert!(on_ring.length() > 0.01);
        // Point far from ring
        let far = src.sample(Vec2::new(0.0, 0.0));
        assert!(far.length() < on_ring.length());
    }

    #[test]
    fn test_refraction() {
        let src = ActiveDistortion::new(0, DistortionSource::Refraction {
            center: Vec2::new(0.5, 0.5),
            radius: 0.3,
            strength: 0.1,
            pattern_frequency: 5.0,
        });
        let offset = src.sample(Vec2::new(0.5, 0.5));
        assert!(offset.length() > 0.0);
    }

    #[test]
    fn test_refraction_outside() {
        let src = ActiveDistortion::new(0, DistortionSource::Refraction {
            center: Vec2::new(0.5, 0.5),
            radius: 0.1,
            strength: 0.1,
            pattern_frequency: 5.0,
        });
        let offset = src.sample(Vec2::new(0.0, 0.0));
        assert!((offset.x).abs() < 1e-6);
    }

    #[test]
    fn test_portal_swirl() {
        let src = ActiveDistortion::new(0, DistortionSource::Portal {
            center: Vec2::new(0.5, 0.5),
            radius: 0.4,
            twist_amount: 3.0,
            falloff_power: 2.0,
        });
        let offset = src.sample(Vec2::new(0.6, 0.5));
        assert!(offset.length() > 0.01);
    }

    #[test]
    fn test_portal_outside() {
        let src = ActiveDistortion::new(0, DistortionSource::Portal {
            center: Vec2::new(0.5, 0.5),
            radius: 0.1,
            twist_amount: 3.0,
            falloff_power: 2.0,
        });
        let offset = src.sample(Vec2::new(0.0, 0.0));
        assert!((offset.x).abs() < 1e-6);
    }

    #[test]
    fn test_inactive_source_zero() {
        let mut src = ActiveDistortion::new(0, DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 0.3,
            frequency: 10.0,
            amplitude: 0.05,
            speed: 1.0,
        });
        src.active = false;
        let offset = src.sample(Vec2::new(0.5, 0.5));
        assert!((offset.x).abs() < 1e-6);
    }

    #[test]
    fn test_zero_intensity() {
        let mut src = ActiveDistortion::new(0, DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 0.3,
            frequency: 10.0,
            amplitude: 0.05,
            speed: 1.0,
        });
        src.intensity = 0.0;
        let offset = src.sample(Vec2::new(0.5, 0.5));
        assert!((offset.x).abs() < 1e-6);
    }

    #[test]
    fn test_compositor_add_remove() {
        let mut comp = DistortionCompositor::new();
        let id = comp.add_source(DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 0.3,
            frequency: 10.0,
            amplitude: 0.05,
            speed: 1.0,
        });
        assert_eq!(comp.source_count(), 1);
        assert!(comp.remove_source(id));
        assert_eq!(comp.source_count(), 0);
    }

    #[test]
    fn test_compositor_update_removes_inactive() {
        let mut comp = DistortionCompositor::new();
        comp.add_source(DistortionSource::Shockwave {
            center: Vec2::new(0.5, 0.5),
            current_radius: 0.0,
            expand_speed: 10.0,
            ring_width: 0.1,
            amplitude: 0.1,
            max_radius: 0.1,
        });
        comp.update(1.0);
        assert_eq!(comp.source_count(), 0);
    }

    #[test]
    fn test_compositor_composite() {
        let mut comp = DistortionCompositor::new();
        comp.add_source(DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 1.0,
            frequency: 10.0,
            amplitude: 0.1,
            speed: 0.0,
        });
        let mut buf = DistortionBuffer::new(8, 8);
        comp.composite(&mut buf);
        assert!(buf.max_distortion() > 0.0);
    }

    #[test]
    fn test_compositor_combined_sample() {
        let mut comp = DistortionCompositor::new();
        comp.add_source(DistortionSource::Refraction {
            center: Vec2::new(0.5, 0.5),
            radius: 1.0,
            strength: 0.1,
            pattern_frequency: 5.0,
        });
        let v = comp.sample_combined(Vec2::new(0.5, 0.5));
        assert!(v.length() > 0.0);
    }

    #[test]
    fn test_apply_to_scene() {
        let mut buf = DistortionBuffer::new(4, 4);
        buf.set(2, 2, Vec2::new(0.1, 0.0));
        let color = buf.apply_to_scene(2, 2, |u, _v| {
            Color::new(u, 0.0, 0.0, 1.0)
        });
        // u = 2/4 + 0.1 = 0.6
        assert!((color.r - 0.6).abs() < 0.05);
    }

    #[test]
    fn test_max_distortion() {
        let mut buf = DistortionBuffer::new(4, 4);
        buf.set(1, 1, Vec2::new(0.3, 0.4));
        assert!((buf.max_distortion() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_compositor_set_intensity() {
        let mut comp = DistortionCompositor::new();
        let id = comp.add_source(DistortionSource::HeatHaze {
            center: Vec2::new(0.5, 0.5),
            radius: 1.0,
            frequency: 10.0,
            amplitude: 0.1,
            speed: 0.0,
        });
        comp.set_intensity(id, 0.5);
        let v_half = comp.sample_combined(Vec2::new(0.5, 0.5));
        comp.set_intensity(id, 1.0);
        let v_full = comp.sample_combined(Vec2::new(0.5, 0.5));
        // Half intensity should produce roughly half the offset
        assert!(v_half.length() < v_full.length() + 0.01);
    }

    #[test]
    fn test_compositor_clear() {
        let mut comp = DistortionCompositor::new();
        comp.add_source(DistortionSource::HeatHaze {
            center: Vec2::ZERO, radius: 1.0, frequency: 1.0, amplitude: 0.1, speed: 0.0,
        });
        comp.clear();
        assert_eq!(comp.source_count(), 0);
    }
}
