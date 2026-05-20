// motion_blur.rs — Per-object and camera motion blur for the rendering pipeline.
//
// Implements velocity buffer generation, gather blur along velocity vectors,
// tile-based early-out for static regions, depth-aware blur, and camera-only mode.

use std::fmt;

/// 2D vector for screen-space velocity.
#[derive(Clone, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn zero() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { x: self.x * s, y: self.y * s }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }

    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            Self::zero()
        } else {
            self.scale(1.0 / len)
        }
    }
}

/// RGBA color pixel.
#[derive(Clone, Debug, PartialEq)]
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

    pub fn black() -> Self {
        Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }

    pub fn scale(&self, s: f64) -> Self {
        Self { r: self.r * s, g: self.g * s, b: self.b * s, a: self.a }
    }

    pub fn add_weighted(&self, other: &Self, w: f64) -> Self {
        Self {
            r: self.r + other.r * w,
            g: self.g + other.g * w,
            b: self.b + other.b * w,
            a: self.a,
        }
    }
}

/// 4x4 matrix for projection/view transforms.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat4 {
    pub m: [f64; 16],
}

impl Mat4 {
    pub fn identity() -> Self {
        let mut m = [0.0; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Self { m }
    }

    pub fn translation(tx: f64, ty: f64, tz: f64) -> Self {
        let mut mat = Self::identity();
        mat.m[12] = tx;
        mat.m[13] = ty;
        mat.m[14] = tz;
        mat
    }

    /// Transform a 3D point (homogeneous divide).
    pub fn transform_point(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let w = self.m[3] * x + self.m[7] * y + self.m[11] * z + self.m[15];
        let safe_w = if w.abs() < 1e-10 { 1e-10 } else { w };
        (
            (self.m[0] * x + self.m[4] * y + self.m[8] * z + self.m[12]) / safe_w,
            (self.m[1] * x + self.m[5] * y + self.m[9] * z + self.m[13]) / safe_w,
            (self.m[2] * x + self.m[6] * y + self.m[10] * z + self.m[14]) / safe_w,
        )
    }

    /// Multiply two 4x4 matrices.
    pub fn mul(&self, other: &Self) -> Self {
        let mut r = [0.0; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum += self.m[row + k * 4] * other.m[k + col * 4];
                }
                r[row + col * 4] = sum;
            }
        }
        Self { m: r }
    }
}

/// 2D buffer of screen-space velocity vectors.
#[derive(Clone, Debug)]
pub struct VelocityBuffer {
    pub width: usize,
    pub height: usize,
    pub velocities: Vec<Vec2>,
}

impl VelocityBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            velocities: vec![Vec2::zero(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &Vec2 {
        &self.velocities[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, v: Vec2) {
        self.velocities[y * self.width + x] = v;
    }
}

/// Color buffer (frame buffer).
#[derive(Clone, Debug)]
pub struct ColorBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Color>,
}

impl ColorBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Color::black(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &Color {
        &self.pixels[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, c: Color) {
        self.pixels[y * self.width + x] = c;
    }

    pub fn sample_clamp(&self, x: isize, y: isize) -> &Color {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }
}

/// Depth buffer.
#[derive(Clone, Debug)]
pub struct DepthBuffer {
    pub width: usize,
    pub height: usize,
    pub depths: Vec<f64>,
}

impl DepthBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            depths: vec![1.0; width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f64 {
        self.depths[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, d: f64) {
        self.depths[y * self.width + x] = d;
    }

    pub fn sample_clamp(&self, x: isize, y: isize) -> f64 {
        let cx = x.max(0).min(self.width as isize - 1) as usize;
        let cy = y.max(0).min(self.height as isize - 1) as usize;
        self.get(cx, cy)
    }
}

/// Motion blur configuration.
#[derive(Clone, Debug)]
pub struct MotionBlurConfig {
    /// Number of samples along the velocity vector (8-16 typical).
    pub sample_count: usize,
    /// Overall motion blur intensity (scales velocity).
    pub intensity: f64,
    /// Tile size for early-out optimization (pixels).
    pub tile_size: usize,
    /// Minimum velocity magnitude to trigger blur.
    pub velocity_threshold: f64,
    /// Depth discontinuity threshold for depth-aware blur.
    pub depth_threshold: f64,
    /// Whether to use camera-only mode (ignore per-object velocity).
    pub camera_only: bool,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            sample_count: 12,
            intensity: 1.0,
            tile_size: 16,
            velocity_threshold: 0.5,
            depth_threshold: 0.05,
            camera_only: false,
        }
    }
}

impl fmt::Display for MotionBlurConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MotionBlurConfig(samples={}, intensity={:.2})", self.sample_count, self.intensity)
    }
}

/// Compute screen-space velocity from current and previous view-projection matrices.
/// Projects pixel position through inverse-current and forward-previous to get motion.
pub fn compute_camera_velocity(
    x: usize,
    y: usize,
    depth: f64,
    width: usize,
    height: usize,
    current_vp: &Mat4,
    prev_vp: &Mat4,
) -> Vec2 {
    // Current NDC position
    let ndc_x = (x as f64 + 0.5) / width as f64 * 2.0 - 1.0;
    let ndc_y = (y as f64 + 0.5) / height as f64 * 2.0 - 1.0;

    // Current world position (approximate via the matrices)
    let curr_pos = current_vp.transform_point(ndc_x, ndc_y, depth);
    // Previous screen position
    let prev_pos = prev_vp.transform_point(curr_pos.0, curr_pos.1, curr_pos.2);

    Vec2::new(
        (ndc_x - prev_pos.0) * 0.5 * width as f64,
        (ndc_y - prev_pos.1) * 0.5 * height as f64,
    )
}

/// Generate velocity buffer from camera motion (camera-only mode).
pub fn generate_camera_velocity_buffer(
    depth_buf: &DepthBuffer,
    current_vp: &Mat4,
    prev_vp: &Mat4,
) -> VelocityBuffer {
    let w = depth_buf.width;
    let h = depth_buf.height;
    let mut vbuf = VelocityBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let d = depth_buf.get(x, y);
            let vel = compute_camera_velocity(x, y, d, w, h, current_vp, prev_vp);
            vbuf.set(x, y, vel);
        }
    }
    vbuf
}

/// Tile-based max velocity computation for early-out optimization.
/// Returns a grid of max velocity magnitudes per tile.
pub struct TileGrid {
    pub tiles_x: usize,
    pub tiles_y: usize,
    pub max_velocities: Vec<f64>,
}

impl TileGrid {
    pub fn from_velocity_buffer(vbuf: &VelocityBuffer, tile_size: usize) -> Self {
        let tiles_x = (vbuf.width + tile_size - 1) / tile_size;
        let tiles_y = (vbuf.height + tile_size - 1) / tile_size;
        let mut max_vel = vec![0.0f64; tiles_x * tiles_y];

        for y in 0..vbuf.height {
            for x in 0..vbuf.width {
                let tx = x / tile_size;
                let ty = y / tile_size;
                let vel_mag = vbuf.get(x, y).length();
                let idx = ty * tiles_x + tx;
                if vel_mag > max_vel[idx] {
                    max_vel[idx] = vel_mag;
                }
            }
        }

        Self {
            tiles_x,
            tiles_y,
            max_velocities: max_vel,
        }
    }

    pub fn get_tile_max(&self, tx: usize, ty: usize) -> f64 {
        self.max_velocities[ty * self.tiles_x + tx]
    }

    /// Check if a tile is effectively static (below threshold).
    pub fn is_static(&self, tx: usize, ty: usize, threshold: f64) -> bool {
        self.get_tile_max(tx, ty) < threshold
    }
}

/// Gather blur along velocity vector at a single pixel.
/// Depth-aware: rejects samples with large depth discontinuity.
fn gather_blur_pixel(
    x: usize,
    y: usize,
    color_buf: &ColorBuffer,
    vel_buf: &VelocityBuffer,
    depth_buf: &DepthBuffer,
    config: &MotionBlurConfig,
) -> Color {
    let vel = vel_buf.get(x, y).scale(config.intensity);
    let vel_len = vel.length();

    if vel_len < config.velocity_threshold {
        return color_buf.get(x, y).clone();
    }

    let center_depth = depth_buf.get(x, y);
    let center_color = color_buf.get(x, y);
    let samples = config.sample_count.max(2);
    let step = vel.scale(1.0 / samples as f64);

    let mut acc = center_color.clone();
    let mut weight = 1.0_f64;

    for i in 1..samples {
        let t = i as f64 - (samples as f64 - 1.0) * 0.5;
        let sx = x as f64 + step.x * t;
        let sy = y as f64 + step.y * t;
        let sxi = sx.round() as isize;
        let syi = sy.round() as isize;

        let sample_depth = depth_buf.sample_clamp(sxi, syi);
        let depth_diff = (sample_depth - center_depth).abs();

        if depth_diff < config.depth_threshold {
            let sample_color = color_buf.sample_clamp(sxi, syi);
            acc = acc.add_weighted(sample_color, 1.0);
            weight += 1.0;
        }
    }

    if weight > 1e-10 {
        acc.scale(1.0 / weight)
    } else {
        center_color.clone()
    }
}

/// Apply motion blur to an entire frame.
pub fn apply_motion_blur(
    color_buf: &ColorBuffer,
    vel_buf: &VelocityBuffer,
    depth_buf: &DepthBuffer,
    config: &MotionBlurConfig,
) -> ColorBuffer {
    let w = color_buf.width;
    let h = color_buf.height;
    let tile_grid = TileGrid::from_velocity_buffer(vel_buf, config.tile_size);
    let mut out = ColorBuffer::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let tx = x / config.tile_size;
            let ty = y / config.tile_size;

            // Early out for static tiles
            if tile_grid.is_static(tx, ty, config.velocity_threshold) {
                out.set(x, y, color_buf.get(x, y).clone());
            } else {
                let blurred = gather_blur_pixel(x, y, color_buf, vel_buf, depth_buf, config);
                out.set(x, y, blurred);
            }
        }
    }
    out
}

/// Apply camera-only motion blur using previous and current view-projection matrices.
pub fn apply_camera_motion_blur(
    color_buf: &ColorBuffer,
    depth_buf: &DepthBuffer,
    current_vp: &Mat4,
    prev_vp: &Mat4,
    config: &MotionBlurConfig,
) -> ColorBuffer {
    let vel_buf = generate_camera_velocity_buffer(depth_buf, current_vp, prev_vp);
    apply_motion_blur(color_buf, &vel_buf, depth_buf, config)
}

/// Compute velocity buffer statistics (for diagnostics).
pub struct VelocityStats {
    pub min_magnitude: f64,
    pub max_magnitude: f64,
    pub avg_magnitude: f64,
    pub static_pixel_count: usize,
    pub moving_pixel_count: usize,
}

impl fmt::Display for VelocityStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VelocityStats(min={:.4}, max={:.4}, avg={:.4}, static={}, moving={})",
            self.min_magnitude, self.max_magnitude, self.avg_magnitude,
            self.static_pixel_count, self.moving_pixel_count
        )
    }
}

pub fn compute_velocity_stats(vbuf: &VelocityBuffer, threshold: f64) -> VelocityStats {
    let mut min_mag = f64::MAX;
    let mut max_mag = 0.0_f64;
    let mut total_mag = 0.0_f64;
    let mut static_count = 0usize;
    let mut moving_count = 0usize;

    for v in &vbuf.velocities {
        let mag = v.length();
        if mag < min_mag { min_mag = mag; }
        if mag > max_mag { max_mag = mag; }
        total_mag += mag;
        if mag < threshold {
            static_count += 1;
        } else {
            moving_count += 1;
        }
    }

    let count = vbuf.velocities.len().max(1);
    VelocityStats {
        min_magnitude: if min_mag == f64::MAX { 0.0 } else { min_mag },
        max_magnitude: max_mag,
        avg_magnitude: total_mag / count as f64,
        static_pixel_count: static_count,
        moving_pixel_count: moving_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!(approx_eq(v.length(), 5.0, 1e-6));
    }

    #[test]
    fn test_vec2_zero_length() {
        assert!(approx_eq(Vec2::zero().length(), 0.0, 1e-10));
    }

    #[test]
    fn test_vec2_normalize() {
        let v = Vec2::new(3.0, 4.0).normalized();
        assert!(approx_eq(v.length(), 1.0, 1e-6));
    }

    #[test]
    fn test_vec2_normalize_zero() {
        let v = Vec2::zero().normalized();
        assert!(approx_eq(v.length(), 0.0, 1e-10));
    }

    #[test]
    fn test_mat4_identity_transform() {
        let m = Mat4::identity();
        let (x, y, z) = m.transform_point(1.0, 2.0, 3.0);
        assert!(approx_eq(x, 1.0, 1e-6));
        assert!(approx_eq(y, 2.0, 1e-6));
        assert!(approx_eq(z, 3.0, 1e-6));
    }

    #[test]
    fn test_mat4_translation() {
        let m = Mat4::translation(10.0, 20.0, 30.0);
        let (x, y, z) = m.transform_point(1.0, 2.0, 3.0);
        assert!(approx_eq(x, 11.0, 1e-6));
        assert!(approx_eq(y, 22.0, 1e-6));
        assert!(approx_eq(z, 33.0, 1e-6));
    }

    #[test]
    fn test_velocity_buffer_set_get() {
        let mut vbuf = VelocityBuffer::new(4, 4);
        vbuf.set(2, 3, Vec2::new(5.0, 10.0));
        let v = vbuf.get(2, 3);
        assert!(approx_eq(v.x, 5.0, 1e-6));
        assert!(approx_eq(v.y, 10.0, 1e-6));
    }

    #[test]
    fn test_color_buffer_clamp() {
        let mut cbuf = ColorBuffer::new(4, 4);
        cbuf.set(0, 0, Color::new(1.0, 0.0, 0.0, 1.0));
        let c = cbuf.sample_clamp(-5, -5);
        assert!(approx_eq(c.r, 1.0, 1e-6));
    }

    #[test]
    fn test_tile_grid_static_tiles() {
        let vbuf = VelocityBuffer::new(16, 16); // all zero velocity
        let grid = TileGrid::from_velocity_buffer(&vbuf, 8);
        assert!(grid.is_static(0, 0, 0.5));
        assert!(grid.is_static(1, 1, 0.5));
    }

    #[test]
    fn test_tile_grid_moving_tile() {
        let mut vbuf = VelocityBuffer::new(16, 16);
        vbuf.set(5, 5, Vec2::new(10.0, 0.0));
        let grid = TileGrid::from_velocity_buffer(&vbuf, 8);
        assert!(!grid.is_static(0, 0, 0.5));
    }

    #[test]
    fn test_motion_blur_static_frame() {
        let cbuf = ColorBuffer::new(8, 8);
        let vbuf = VelocityBuffer::new(8, 8);
        let dbuf = DepthBuffer::new(8, 8);
        let config = MotionBlurConfig::default();
        let result = apply_motion_blur(&cbuf, &vbuf, &dbuf, &config);
        // Static frame, output equals input
        let px = result.get(4, 4);
        assert!(approx_eq(px.r, 0.0, 1e-6));
    }

    #[test]
    fn test_motion_blur_with_velocity() {
        let mut cbuf = ColorBuffer::new(16, 16);
        let mut vbuf = VelocityBuffer::new(16, 16);
        let dbuf = DepthBuffer::new(16, 16);

        // Create a horizontal stripe of color with velocity
        for x in 0..16 {
            cbuf.set(x, 8, Color::new(1.0, 0.0, 0.0, 1.0));
            vbuf.set(x, 8, Vec2::new(3.0, 0.0));
        }

        let config = MotionBlurConfig {
            sample_count: 8,
            intensity: 1.0,
            tile_size: 8,
            velocity_threshold: 0.5,
            depth_threshold: 1.0,
            camera_only: false,
        };
        let result = apply_motion_blur(&cbuf, &vbuf, &dbuf, &config);
        // Center pixel should still have some red
        let px = result.get(8, 8);
        assert!(px.r > 0.0);
    }

    #[test]
    fn test_depth_aware_blur_rejects_discontinuity() {
        let mut cbuf = ColorBuffer::new(8, 8);
        let mut vbuf = VelocityBuffer::new(8, 8);
        let mut dbuf = DepthBuffer::new(8, 8);

        cbuf.set(4, 4, Color::new(1.0, 1.0, 1.0, 1.0));
        vbuf.set(4, 4, Vec2::new(2.0, 0.0));
        dbuf.set(4, 4, 0.1);
        // Neighbor at very different depth
        dbuf.set(5, 4, 0.9);
        cbuf.set(5, 4, Color::new(0.0, 0.0, 10.0, 1.0));

        let config = MotionBlurConfig {
            depth_threshold: 0.05,
            sample_count: 4,
            ..Default::default()
        };
        let result = apply_motion_blur(&cbuf, &vbuf, &dbuf, &config);
        // The blue bleed should be suppressed
        let px = result.get(4, 4);
        assert!(px.b < 5.0);
    }

    #[test]
    fn test_camera_velocity_stationary() {
        let vp = Mat4::identity();
        let vel = compute_camera_velocity(4, 4, 0.5, 8, 8, &vp, &vp);
        // No camera movement means no velocity
        assert!(vel.length() < 1e-6);
    }

    #[test]
    fn test_camera_motion_blur_pipeline() {
        let cbuf = ColorBuffer::new(8, 8);
        let dbuf = DepthBuffer::new(8, 8);
        let current = Mat4::identity();
        let prev = Mat4::identity();
        let config = MotionBlurConfig::default();
        let result = apply_camera_motion_blur(&cbuf, &dbuf, &current, &prev, &config);
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_velocity_stats_all_static() {
        let vbuf = VelocityBuffer::new(4, 4);
        let stats = compute_velocity_stats(&vbuf, 0.5);
        assert_eq!(stats.static_pixel_count, 16);
        assert_eq!(stats.moving_pixel_count, 0);
        assert!(approx_eq(stats.avg_magnitude, 0.0, 1e-10));
    }

    #[test]
    fn test_velocity_stats_mixed() {
        let mut vbuf = VelocityBuffer::new(4, 4);
        vbuf.set(0, 0, Vec2::new(5.0, 0.0));
        vbuf.set(1, 1, Vec2::new(0.0, 3.0));
        let stats = compute_velocity_stats(&vbuf, 0.5);
        assert_eq!(stats.moving_pixel_count, 2);
        assert_eq!(stats.static_pixel_count, 14);
        assert!(approx_eq(stats.max_magnitude, 5.0, 1e-6));
    }

    #[test]
    fn test_velocity_stats_display() {
        let vbuf = VelocityBuffer::new(2, 2);
        let stats = compute_velocity_stats(&vbuf, 0.5);
        let s = format!("{}", stats);
        assert!(s.contains("VelocityStats"));
    }

    #[test]
    fn test_motion_blur_config_display() {
        let config = MotionBlurConfig::default();
        let s = format!("{}", config);
        assert!(s.contains("MotionBlurConfig"));
        assert!(s.contains("samples=12"));
    }

    #[test]
    fn test_generate_camera_velocity_buffer() {
        let dbuf = DepthBuffer::new(4, 4);
        let cur = Mat4::identity();
        let prev = Mat4::identity();
        let vbuf = generate_camera_velocity_buffer(&dbuf, &cur, &prev);
        assert_eq!(vbuf.width, 4);
        assert_eq!(vbuf.height, 4);
        for v in &vbuf.velocities {
            assert!(v.length() < 1e-4);
        }
    }

    #[test]
    fn test_color_add_weighted() {
        let a = Color::new(1.0, 0.0, 0.0, 1.0);
        let b = Color::new(0.0, 2.0, 0.0, 1.0);
        let c = a.add_weighted(&b, 0.5);
        assert!(approx_eq(c.r, 1.0, 1e-6));
        assert!(approx_eq(c.g, 1.0, 1e-6));
    }

    #[test]
    fn test_mat4_multiply_identity() {
        let a = Mat4::identity();
        let b = Mat4::translation(1.0, 2.0, 3.0);
        let c = a.mul(&b);
        let (x, y, z) = c.transform_point(0.0, 0.0, 0.0);
        assert!(approx_eq(x, 1.0, 1e-6));
        assert!(approx_eq(y, 2.0, 1e-6));
        assert!(approx_eq(z, 3.0, 1e-6));
    }

    #[test]
    fn test_tile_grid_dimensions() {
        let vbuf = VelocityBuffer::new(17, 9);
        let grid = TileGrid::from_velocity_buffer(&vbuf, 8);
        assert_eq!(grid.tiles_x, 3); // ceil(17/8)
        assert_eq!(grid.tiles_y, 2); // ceil(9/8)
    }
}
