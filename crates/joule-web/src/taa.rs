// taa.rs — Temporal Anti-Aliasing
// Jittered projection, history reprojection, neighborhood clamping, velocity rejection.

/// RGBA pixel as floats for high-precision blending.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorF {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl ColorF {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn black() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    pub fn luminance(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    pub fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        Self {
            r: a.r + (b.r - a.r) * t,
            g: a.g + (b.g - a.g) * t,
            b: a.b + (b.b - a.b) * t,
            a: a.a + (b.a - a.a) * t,
        }
    }

    pub fn clamp_components(&self, lo: &Self, hi: &Self) -> Self {
        Self {
            r: self.r.clamp(lo.r, hi.r),
            g: self.g.clamp(lo.g, hi.g),
            b: self.b.clamp(lo.b, hi.b),
            a: self.a.clamp(lo.a, hi.a),
        }
    }

    pub fn component_min(a: &Self, b: &Self) -> Self {
        Self {
            r: a.r.min(b.r),
            g: a.g.min(b.g),
            b: a.b.min(b.b),
            a: a.a.min(b.a),
        }
    }

    pub fn component_max(a: &Self, b: &Self) -> Self {
        Self {
            r: a.r.max(b.r),
            g: a.g.max(b.g),
            b: a.b.max(b.b),
            a: a.a.max(b.a),
        }
    }
}

/// 2D motion vector per pixel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    pub dx: f32,
    pub dy: f32,
}

impl MotionVector {
    pub fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }

    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    pub fn length(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }
}

/// 2D sub-pixel jitter offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JitterOffset {
    pub x: f32,
    pub y: f32,
}

/// Generate a Halton sequence value for a given index and base.
pub fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let mut i = index;
    let bf = base as f32;
    while i > 0 {
        f /= bf;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// Generate a jitter offset for a given frame index using Halton(2,3) sequence.
/// Returns offset in [-0.5, 0.5) range.
pub fn jitter_for_frame(frame_index: u32) -> JitterOffset {
    let idx = frame_index.max(1);
    JitterOffset {
        x: halton(idx, 2) - 0.5,
        y: halton(idx, 3) - 0.5,
    }
}

/// Generate a full jitter sequence for N frames.
pub fn jitter_sequence(count: u32) -> Vec<JitterOffset> {
    (1..=count).map(|i| jitter_for_frame(i)).collect()
}

/// Frame buffer of floating-point colors.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameBuffer {
    pub pixels: Vec<ColorF>,
    pub width: usize,
    pub height: usize,
}

impl FrameBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![ColorF::black(); width * height],
            width,
            height,
        }
    }

    pub fn from_pixels(pixels: Vec<ColorF>, width: usize, height: usize) -> Option<Self> {
        if pixels.len() != width * height {
            return None;
        }
        Some(Self { pixels, width, height })
    }

    pub fn get(&self, x: usize, y: usize) -> ColorF {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x]
        } else {
            ColorF::black()
        }
    }

    pub fn get_bilinear(&self, u: f32, v: f32) -> ColorF {
        let fx = u * self.width as f32 - 0.5;
        let fy = v * self.height as f32 - 0.5;

        let x0 = fx.floor() as isize;
        let y0 = fy.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let frac_x = fx - fx.floor();
        let frac_y = fy - fy.floor();

        let sample = |sx: isize, sy: isize| -> ColorF {
            let cx = sx.clamp(0, self.width as isize - 1) as usize;
            let cy = sy.clamp(0, self.height as isize - 1) as usize;
            self.pixels[cy * self.width + cx]
        };

        let c00 = sample(x0, y0);
        let c10 = sample(x1, y0);
        let c01 = sample(x0, y1);
        let c11 = sample(x1, y1);

        let top = ColorF::lerp(&c00, &c10, frac_x);
        let bot = ColorF::lerp(&c01, &c11, frac_x);
        ColorF::lerp(&top, &bot, frac_y)
    }

    pub fn set(&mut self, x: usize, y: usize, color: ColorF) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = color;
        }
    }
}

/// Motion vector buffer (same dimensions as frame).
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBuffer {
    pub vectors: Vec<MotionVector>,
    pub width: usize,
    pub height: usize,
}

impl MotionBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            vectors: vec![MotionVector::zero(); width * height],
            width,
            height,
        }
    }

    pub fn get(&self, x: usize, y: usize) -> MotionVector {
        if x < self.width && y < self.height {
            self.vectors[y * self.width + x]
        } else {
            MotionVector::zero()
        }
    }

    pub fn set(&mut self, x: usize, y: usize, mv: MotionVector) {
        if x < self.width && y < self.height {
            self.vectors[y * self.width + x] = mv;
        }
    }
}

/// TAA configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaaConfig {
    /// Blend factor for current frame (typically 0.05 - 0.2).
    pub current_weight: f32,
    /// Enable neighborhood clamping to reduce ghosting.
    pub neighborhood_clamp: bool,
    /// Enable velocity-based rejection weighting.
    pub velocity_rejection: bool,
    /// Velocity threshold above which history is rejected (in pixels/frame).
    pub velocity_reject_threshold: f32,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            current_weight: 0.1,
            neighborhood_clamp: true,
            velocity_rejection: true,
            velocity_reject_threshold: 4.0,
        }
    }
}

/// Compute the AABB (axis-aligned bounding box) of the 3x3 neighborhood for clamping.
fn neighborhood_aabb(
    current: &FrameBuffer,
    x: usize,
    y: usize,
) -> (ColorF, ColorF) {
    let mut lo = ColorF::new(f32::MAX, f32::MAX, f32::MAX, f32::MAX);
    let mut hi = ColorF::new(f32::MIN, f32::MIN, f32::MIN, f32::MIN);

    for dy in 0..3usize {
        for dx in 0..3usize {
            let sx = if x + dx >= 1 { x + dx - 1 } else { 0 };
            let sy = if y + dy >= 1 { y + dy - 1 } else { 0 };
            let c = current.get(sx.min(current.width - 1), sy.min(current.height - 1));
            lo = ColorF::component_min(&lo, &c);
            hi = ColorF::component_max(&hi, &c);
        }
    }

    (lo, hi)
}

/// Reproject a pixel coordinate using a motion vector to get history UV.
pub fn reproject(x: usize, y: usize, mv: &MotionVector, width: usize, height: usize) -> (f32, f32) {
    let u = (x as f32 + 0.5 - mv.dx) / width as f32;
    let v = (y as f32 + 0.5 - mv.dy) / height as f32;
    (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0))
}

/// Compute velocity-based rejection weight. Returns 0.0-1.0 where 1.0 = fully trust history.
pub fn velocity_rejection_weight(mv: &MotionVector, threshold: f32) -> f32 {
    let speed = mv.length();
    if speed < 1e-6 {
        return 1.0;
    }
    (1.0 - speed / threshold).clamp(0.0, 1.0)
}

/// Resolve a single TAA pixel.
pub fn taa_resolve_pixel(
    current: &FrameBuffer,
    history: &FrameBuffer,
    motion: &MotionBuffer,
    x: usize,
    y: usize,
    config: &TaaConfig,
) -> ColorF {
    let current_color = current.get(x, y);
    let mv = motion.get(x, y);

    // Reproject to history.
    let (hu, hv) = reproject(x, y, &mv, current.width, current.height);
    let mut history_color = history.get_bilinear(hu, hv);

    // Neighborhood clamping.
    if config.neighborhood_clamp {
        let (lo, hi) = neighborhood_aabb(current, x, y);
        history_color = history_color.clamp_components(&lo, &hi);
    }

    // Blend factor adjustment.
    let mut blend = config.current_weight;

    // Velocity rejection.
    if config.velocity_rejection {
        let trust = velocity_rejection_weight(&mv, config.velocity_reject_threshold);
        // Higher velocity = less history trust = more current weight.
        blend = blend + (1.0 - trust) * (1.0 - blend);
    }

    blend = blend.clamp(0.0, 1.0);
    ColorF::lerp(&history_color, &current_color, blend)
}

/// Apply TAA to an entire frame, producing the resolved output.
pub fn apply_taa(
    current: &FrameBuffer,
    history: &FrameBuffer,
    motion: &MotionBuffer,
    config: &TaaConfig,
) -> FrameBuffer {
    let w = current.width;
    let h = current.height;
    let mut result = FrameBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            result.set(x, y, taa_resolve_pixel(current, history, motion, x, y, config));
        }
    }

    result
}

/// TAA state machine that accumulates frames.
#[derive(Debug, Clone)]
pub struct TaaAccumulator {
    pub history: Option<FrameBuffer>,
    pub frame_count: u32,
    pub config: TaaConfig,
}

impl TaaAccumulator {
    pub fn new(config: TaaConfig) -> Self {
        Self {
            history: None,
            frame_count: 0,
            config,
        }
    }

    /// Get the jitter offset for the next frame.
    pub fn next_jitter(&self) -> JitterOffset {
        jitter_for_frame(self.frame_count + 1)
    }

    /// Process a new frame, returning the resolved result.
    pub fn accumulate(
        &mut self,
        current: &FrameBuffer,
        motion: &MotionBuffer,
    ) -> FrameBuffer {
        self.frame_count += 1;

        let result = match &self.history {
            Some(hist) => apply_taa(current, hist, motion, &self.config),
            None => current.clone(),
        };

        self.history = Some(result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_frame(w: usize, h: usize, c: ColorF) -> FrameBuffer {
        FrameBuffer {
            pixels: vec![c; w * h],
            width: w,
            height: h,
        }
    }

    fn zero_motion(w: usize, h: usize) -> MotionBuffer {
        MotionBuffer::new(w, h)
    }

    #[test]
    fn test_halton_base2() {
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_halton_base3() {
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_halton_zero() {
        assert!(halton(0, 2).abs() < 1e-6);
    }

    #[test]
    fn test_jitter_range() {
        for i in 1..=64 {
            let j = jitter_for_frame(i);
            assert!(j.x >= -0.5 && j.x <= 0.5, "x out of range: {}", j.x);
            assert!(j.y >= -0.5 && j.y <= 0.5, "y out of range: {}", j.y);
        }
    }

    #[test]
    fn test_jitter_sequence_length() {
        let seq = jitter_sequence(16);
        assert_eq!(seq.len(), 16);
    }

    #[test]
    fn test_jitter_sequence_varies() {
        let seq = jitter_sequence(8);
        let first = seq[0];
        let all_same = seq.iter().all(|j| (j.x - first.x).abs() < 1e-6 && (j.y - first.y).abs() < 1e-6);
        assert!(!all_same, "jitter sequence should vary");
    }

    #[test]
    fn test_colorf_luminance() {
        let white = ColorF::new(1.0, 1.0, 1.0, 1.0);
        assert!((white.luminance() - 1.0).abs() < 1e-3);

        let black = ColorF::black();
        assert!(black.luminance().abs() < 1e-6);
    }

    #[test]
    fn test_colorf_lerp() {
        let a = ColorF::new(0.0, 0.0, 0.0, 1.0);
        let b = ColorF::new(1.0, 1.0, 1.0, 1.0);
        let mid = ColorF::lerp(&a, &b, 0.5);
        assert!((mid.r - 0.5).abs() < 1e-6);
        assert!((mid.g - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_colorf_clamp_components() {
        let c = ColorF::new(2.0, -1.0, 0.5, 0.5);
        let lo = ColorF::new(0.0, 0.0, 0.0, 0.0);
        let hi = ColorF::new(1.0, 1.0, 1.0, 1.0);
        let clamped = c.clamp_components(&lo, &hi);
        assert!((clamped.r - 1.0).abs() < 1e-6);
        assert!(clamped.g.abs() < 1e-6);
        assert!((clamped.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_frame_buffer_bilinear() {
        let mut fb = FrameBuffer::new(2, 2);
        fb.set(0, 0, ColorF::new(0.0, 0.0, 0.0, 1.0));
        fb.set(1, 0, ColorF::new(1.0, 0.0, 0.0, 1.0));
        fb.set(0, 1, ColorF::new(0.0, 1.0, 0.0, 1.0));
        fb.set(1, 1, ColorF::new(1.0, 1.0, 0.0, 1.0));

        let center = fb.get_bilinear(0.5, 0.5);
        assert!((center.r - 0.5).abs() < 0.1);
        assert!((center.g - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_motion_vector_length() {
        let mv = MotionVector::new(3.0, 4.0);
        assert!((mv.length() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_reproject_no_motion() {
        let (u, v) = reproject(5, 5, &MotionVector::zero(), 10, 10);
        assert!((u - 0.55).abs() < 1e-6);
        assert!((v - 0.55).abs() < 1e-6);
    }

    #[test]
    fn test_reproject_with_motion() {
        let mv = MotionVector::new(2.0, -1.0);
        let (u, v) = reproject(5, 5, &mv, 10, 10);
        assert!((u - 0.35).abs() < 1e-6);
        assert!((v - 0.65).abs() < 1e-6);
    }

    #[test]
    fn test_reproject_clamps() {
        let mv = MotionVector::new(100.0, 100.0);
        let (u, v) = reproject(0, 0, &mv, 10, 10);
        assert!(u >= 0.0 && u <= 1.0);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_velocity_rejection_zero() {
        let w = velocity_rejection_weight(&MotionVector::zero(), 4.0);
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_velocity_rejection_high_speed() {
        let mv = MotionVector::new(10.0, 0.0);
        let w = velocity_rejection_weight(&mv, 4.0);
        assert!(w.abs() < 1e-6, "high speed should reject history");
    }

    #[test]
    fn test_velocity_rejection_mid_speed() {
        let mv = MotionVector::new(2.0, 0.0);
        let w = velocity_rejection_weight(&mv, 4.0);
        assert!((w - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_taa_static_scene() {
        let w = 8;
        let h = 8;
        let current = solid_frame(w, h, ColorF::new(0.5, 0.5, 0.5, 1.0));
        let history = solid_frame(w, h, ColorF::new(0.5, 0.5, 0.5, 1.0));
        let motion = zero_motion(w, h);
        let config = TaaConfig::default();

        let result = apply_taa(&current, &history, &motion, &config);
        for p in &result.pixels {
            assert!((p.r - 0.5).abs() < 1e-3);
        }
    }

    #[test]
    fn test_taa_blends_history() {
        let w = 4;
        let h = 4;
        let current = solid_frame(w, h, ColorF::new(1.0, 1.0, 1.0, 1.0));
        let history = solid_frame(w, h, ColorF::new(0.0, 0.0, 0.0, 1.0));
        let motion = zero_motion(w, h);
        let config = TaaConfig {
            current_weight: 0.1,
            neighborhood_clamp: false,
            velocity_rejection: false,
            velocity_reject_threshold: 4.0,
        };

        let result = apply_taa(&current, &history, &motion, &config);
        // With 10% current, 90% history: result should be ~0.1.
        let p = result.get(2, 2);
        assert!((p.r - 0.1).abs() < 1e-3);
    }

    #[test]
    fn test_taa_neighborhood_clamp() {
        let w = 4;
        let h = 4;
        let current = solid_frame(w, h, ColorF::new(0.5, 0.5, 0.5, 1.0));
        // History has very different color.
        let history = solid_frame(w, h, ColorF::new(1.0, 0.0, 0.0, 1.0));
        let motion = zero_motion(w, h);
        let config = TaaConfig {
            current_weight: 0.1,
            neighborhood_clamp: true,
            velocity_rejection: false,
            velocity_reject_threshold: 4.0,
        };

        let result = apply_taa(&current, &history, &motion, &config);
        // Neighborhood clamp should pull history toward current range.
        let p = result.get(2, 2);
        assert!((p.r - 0.5).abs() < 0.05, "clamping should constrain history r: {}", p.r);
    }

    #[test]
    fn test_accumulator_first_frame() {
        let config = TaaConfig::default();
        let mut acc = TaaAccumulator::new(config);
        let frame = solid_frame(4, 4, ColorF::new(0.3, 0.3, 0.3, 1.0));
        let motion = zero_motion(4, 4);

        let result = acc.accumulate(&frame, &motion);
        // First frame = no history, should return current.
        let p = result.get(2, 2);
        assert!((p.r - 0.3).abs() < 1e-6);
        assert_eq!(acc.frame_count, 1);
        assert!(acc.history.is_some());
    }

    #[test]
    fn test_accumulator_converges() {
        let config = TaaConfig {
            current_weight: 0.1,
            neighborhood_clamp: false,
            velocity_rejection: false,
            velocity_reject_threshold: 4.0,
        };
        let mut acc = TaaAccumulator::new(config);
        let frame = solid_frame(4, 4, ColorF::new(0.8, 0.8, 0.8, 1.0));
        let motion = zero_motion(4, 4);

        // Accumulate many frames of the same color.
        for _ in 0..50 {
            acc.accumulate(&frame, &motion);
        }

        let result = acc.accumulate(&frame, &motion);
        let p = result.get(2, 2);
        assert!((p.r - 0.8).abs() < 0.01, "should converge to input: {}", p.r);
    }

    #[test]
    fn test_accumulator_jitter() {
        let config = TaaConfig::default();
        let acc = TaaAccumulator::new(config);
        let j = acc.next_jitter();
        assert!(j.x >= -0.5 && j.x <= 0.5);
        assert!(j.y >= -0.5 && j.y <= 0.5);
    }

    #[test]
    fn test_neighborhood_aabb_uniform() {
        let fb = solid_frame(4, 4, ColorF::new(0.5, 0.5, 0.5, 1.0));
        let (lo, hi) = neighborhood_aabb(&fb, 2, 2);
        assert!((lo.r - 0.5).abs() < 1e-6);
        assert!((hi.r - 0.5).abs() < 1e-6);
    }
}
